##########
# THIS CODE IS STILL EXPERIMENTAL AND YOU SHOULDN'T USE IT YET.
########

# IterDataPipe for preprocessing data, tokenizing, and caching to disk
# The general file format we're going with is an apache parquet file with columns for the output of the tokenizer,
# Parquet files are column stores, which means that we can grab token slices from the file and use them easily
# A row is a single doc
# (We might add back in file metadata later)
# We don't want to have one giant file, so we'll split it up into chunks.
import json
import os
from pathlib import Path
from typing import Iterator, Optional

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sprucfluo
from torch.utils.data import IterDataPipe
from tqdm import tqdm
from transformers import BatchEncoding, AutoTokenizer, PreTrainedTokenizerFast

# As a heuristic, we're aiming for files that are around ~250MB
# Typically we're training on sequences of length ~1024 and batch size up to 512, so better to make it divisible by that.
# 4bytes * 512 * 1024 = 2Mi, so we'll go with 128 * 512 * 1024 = 67108864 tokens, which is about 256MiB
from src.corpora.tokenization_utils import batch_tokenize

NUM_TOKENS_PER_FILE = 67108864

# TASKS:
# TODO: only do the caching on local_rank=0 so we do it once per device
# TODO: figure out how to best do multiple nodes
# TODO: make sure we handle reentrancy correctly in the dataset
# TODO: want to also handle being interrupted mid-file, and continuing where we left off.
# TODO: figure out directory structure for caching multiple sources
# TODO: if we're super careful we can compute the number of samples (for a given batch size and stride) in advance
#       if we do that, we can implement a Map-style dataset, which is somewhat preferable when not streaming

LEDGER_FILE = "ledger.json"


class IndexedDataset(IterDataPipe[BatchEncoding]):
    def __init__(self, cache_dir, seq_len: int, stride: Optional[int] = None):
        self.cache_dir = cache_dir
        self.ledger = self._load_ledger()
        self.seq_len = seq_len
        self.stride = stride

    def _files(self):
        for entry in self.ledger["files"]:
            yield entry["file_name"]

    def __iter__(self):
        for file_name in self._files():
            for entry in read_cache_file(file_name, flatten=True):
                yield from sprucfluo.concatenate_and_group_texts(entry, self.seq_len, self.stride)


    @staticmethod
    def build_or_load(token_iter: Iterator[BatchEncoding],
                      cache_dir: str,
                      seq_len: int,
                      stride: Optional[int] = None,
                      num_tokens_per_file: int = NUM_TOKENS_PER_FILE,
                      file_template: str = 'docs-{}.parquet') -> 'IndexedDataset':
        os.makedirs(cache_dir, exist_ok=True)
        ledger_file = os.path.join(cache_dir, LEDGER_FILE)

        if os.path.exists(ledger_file):
            return IndexedDataset(cache_dir, seq_len, stride)

        file_index = 0
        current_writer: Optional[pq.ParquetWriter] = None
        current_num_tokens = 0
        tq: tqdm = tqdm(desc=f"file {file_index} progress", total=num_tokens_per_file, unit="token")
        file_out: Optional[Path] = None

        # list of (file_name, num_tokens), to be output at the end if we finish the whole iterator
        ledger_files = []

        def close_writer():
            nonlocal current_writer, file_out, file_index, current_num_tokens
            if current_writer is not None:
                current_writer.close()
                current_writer = None

            if current_num_tokens > 0:
                ledger_files.append({"file_name": str(file_out), "num_tokens": current_num_tokens})

            file_index += 1
            current_num_tokens = 0

        def reset_writer(schema):
            nonlocal current_writer, tq, file_out, file_index
            file_out = Path(f"{cache_dir}/{file_template.format(file_index)}")
            file_out.parent.mkdir(parents=True, exist_ok=True)
            current_writer = pq.ParquetWriter(file_out, schema, version="2.6", compression="ZSTD")

            tq.reset()
            tq.set_description(f"file {file_index} progress")

        def as_record_batch(doc):
            names, columns = zip(*[(k, pa.array(v)) for k, v in doc.items()])
            return pa.RecordBatch.from_arrays(list(columns), names)

        try:
            for tokens in token_iter:
                batch = as_record_batch(tokens)
                batch_len = sum(len(t) for t in tokens["input_ids"])
                if not current_writer:
                    reset_writer(batch.schema)
                # NB: the elif means we'll write to this file if it's brand new even if the batch is too big
                # TODO: should we maybe split the batch if it's too big?
                elif current_num_tokens + batch_len > num_tokens_per_file:
                    close_writer()
                    reset_writer(batch.schema)

                current_writer.write_batch(batch)
                current_num_tokens += batch_len
                tq.update(batch_len)

            if current_writer:
                tq.reset(current_num_tokens)
                tq.update(current_num_tokens)
                close_writer()

            # if we successfully wrote the whole iterator, we can write the ledger
            with open(ledger_file, "w") as f:
                ledger = {"files": ledger_files}
                json.dump(ledger, f)

            return IndexedDataset(cache_dir, seq_len, stride)
        except (KeyboardInterrupt, InterruptedError):
            current_writer.close()
            current_writer = None
            file_out.unlink(missing_ok=True)
            raise

    def _load_ledger(self):
        ledger_path = os.path.join(self.cache_dir, LEDGER_FILE)
        if os.path.exists(ledger_path):
            with open(ledger_path, "r") as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"{self.cache_dir} is not a complete cache")


def read_cache_file(file, flatten: bool = False) -> Iterator[BatchEncoding]:
    """ Reads the cache files produced by cache_and_group and yields tokenized sequences.
    If flatten is false, this returns the docs as they were presented to the caching process. If flatten is True,
    then the documents returned are actually concatenated documents, where the number is the number of documents
    presented as a batch to the caching process."""
    for b in pq.read_table(file).to_batches():
        if flatten:
            # insert a newaxis to the beginning so that it appears to be bs=1
            yield BatchEncoding(
                {b.field(i).name: b.column(i).values.to_numpy(zero_copy_only=True)[np.newaxis, :] for i in
                 range(b.num_columns)}
            )
        else:
            yield BatchEncoding(
                {b.field(i).name: b.column(i).to_numpy(zero_copy_only=False) for i in range(b.num_columns)})


if __name__ == '__main__':
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained('gpt2')
    dataset = datasets.load_dataset("dlwh/wikitext_2_detokenized", split="train")
    token_iter = batch_tokenize(dataset, tokenizer, batch_size=1000)
    indexer = IndexedDataset.build_or_load(batch_tokenize(dataset, tokenizer, batch_size=1000),
                                           "cache/wikitext-2-indexed", seq_len=512, stride=None)

    for i, batch in enumerate(indexer):
        print(i, batch)

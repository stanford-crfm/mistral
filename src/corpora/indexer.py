# IterDataPipe for preprocessing data, tokenizing, and caching to disk
# The general file format we're going with is an apache parquet file with columns for the output of the tokenizer,
# A row is a single doc. Parquet files are efficient column stores, which means that we can grab token slices from
# multiple docs as a single operation, which makes concatenation much faster (and means we don't need to cache slices).
# (We might add back in file metadata later? though huggingface deletes it)
# We don't want to have one giant file, so we'll split it up into chunks.
# In general, an IndexedDataset is a directory of parquet files plus a metadata file called the ledger.
# The ledger is a json file with the following structure:
# {
#   "files": { "file_name": <name>, "num_tokens": <num_tokens>},
# }
# We don't actually use the num_tokens field, but it's useful for sanity checking.
# The ledger is written last, so we can always check to see if we were interrupted.
import json
import logging
import os
from pathlib import Path
from typing import Iterator, Optional, Union

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


try:
    from torchdata.datapipes.iter import IterDataPipe
except ImportError:
    from torch.utils.data import IterDataPipe

from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast

from src.corpora.tokenization_utils import batch_tokenize, concatenate_and_group_texts


# As a heuristic, we're aiming for files that are around ~250MB
# Typically we're training on sequences of length ~1024 and batch size up to 512, so better to make it divisible by that.
# 4bytes * 512 * 1024 = 2Mi, so we'll go with 128 * 512 * 1024 = 67108864 tokens, which is about 256MiB

NUM_TOKENS_PER_FILE = 67108864

overwatch = logging.getLogger("mistral.corpora.indexer")

# TASKS:
# TODO: figure out directory structure for caching multiple sources
# TODO: if we're super careful we can compute the number of samples (for a given batch size and stride) in advance
#       if we do that, we can implement a Map-style dataset, which is somewhat preferable when not streaming
# TODO: bring in sprucfluo/simultaneous caching and streaming if we want.

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
            for entry in read_cache_file(f"{self.cache_dir}/{file_name}", flatten=True):
                yield from concatenate_and_group_texts(entry, self.seq_len, self.stride)

    @staticmethod
    def build_or_load(
        token_iter: Iterator[BatchEncoding],
        cache_dir: Union[str, os.PathLike],
        seq_len: int,
        stride: Optional[int] = None,
        num_tokens_per_file: int = NUM_TOKENS_PER_FILE,
        file_template: str = "docs-{}.parquet",
    ) -> "IndexedDataset":
        os.makedirs(cache_dir, exist_ok=True)
        ledger_file = os.path.join(cache_dir, LEDGER_FILE)

        if os.path.exists(ledger_file):
            overwatch.info("Found existing indexed dataset at %s", cache_dir)
            return IndexedDataset(cache_dir, seq_len, stride)

        file_index = 0
        current_writer: Optional[pq.ParquetWriter] = None
        current_num_tokens = 0
        tq: tqdm = tqdm(desc=f"file {file_index} progress", total=num_tokens_per_file, unit="token")
        file_out: Optional[str] = None

        # list of (file_name, num_tokens), to be output at the end if we finish the whole iterator
        ledger_files = []

        def close_writer():
            nonlocal current_writer, file_out, file_index, current_num_tokens
            if current_writer is not None:
                current_writer.close()
                current_writer = None

            if current_num_tokens > 0:
                ledger_files.append({"file_name": str(file_out), "num_tokens": current_num_tokens})

        try:
            for tokens in token_iter:
                batch = _as_record_batch(tokens)
                batch_len = sum(len(t) for t in tokens["input_ids"])

                if current_writer and current_num_tokens + batch_len > num_tokens_per_file:
                    close_writer()

                if not current_writer:
                    file_out = file_template.format(file_index)
                    path = Path(f"{cache_dir}/{file_out}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    file_index += 1

                    current_writer = pq.ParquetWriter(path, batch.schema, version="2.6", compression="ZSTD")

                    current_num_tokens = 0

                    tq.reset()
                    tq.set_description(f"file {file_index} progress")

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
            file_out.unlink(missing_ok=True)  # type: ignore
            raise

    def _load_ledger(self):
        ledger_path = os.path.join(self.cache_dir, LEDGER_FILE)
        if os.path.exists(ledger_path):
            with open(ledger_path, "r") as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"{self.cache_dir} is not a complete cache")


def read_cache_file(file, flatten: bool = False) -> Iterator[BatchEncoding]:
    """Reads the cache files produced by cache_and_group and yields tokenized sequences.
    If flatten is false, this returns the docs as they were presented to the caching process. If flatten is True,
    then the documents returned are actually concatenated documents, where the number is the number of documents
    presented as a batch to the caching process."""
    for b in pq.read_table(file).to_batches():
        if flatten:
            # insert a newaxis to the beginning so that it appears to be bs=1
            yield BatchEncoding(
                {
                    b.field(i).name: b.column(i).values.to_numpy(zero_copy_only=True)[np.newaxis, :]
                    for i in range(b.num_columns)
                }
            )
        else:
            yield BatchEncoding(
                {b.field(i).name: b.column(i).to_numpy(zero_copy_only=False) for i in range(b.num_columns)}
            )


def _as_record_batch(doc: BatchEncoding) -> pa.RecordBatch:
    names, columns = zip(*[(k, pa.array(v)) for k, v in doc.items()])
    return pa.RecordBatch.from_arrays(list(columns), names)


if __name__ == "__main__":
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("gpt2")
    dataset = datasets.load_dataset("dlwh/wikitext_2_detokenized", split="train")
    token_iter = batch_tokenize(dataset, tokenizer, batch_size=1000)
    indexer = IndexedDataset.build_or_load(
        batch_tokenize(dataset, tokenizer, batch_size=1000), "cache/wikitext-2-indexed", seq_len=512, stride=None
    )

    for i, batch in enumerate(indexer):
        print(i, batch)

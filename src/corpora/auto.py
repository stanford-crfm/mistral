"""
auto.py

Default Dataset/Corpus Utilities. Downloads (if necessary) from the Hugging Face `datasets` Hub, and organizes into
de-facto training, validation, and testing tests. Performs additional tokenization and normalization as well.
"""
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Callable, Optional
import dill as pickle


import datasets
import torch
from datasets import IterableDatasetDict, IterableDataset, DatasetDict
from datasets.iterable_dataset import iterable_dataset
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerBase

from src.corpora.detokenization import DATASET_TOKENIZATION_REGISTRY


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.corpora.auto")


def get_auto_dataset(
    tokenizer: PreTrainedTokenizer,
    paths: Dict[str, Path],
    dataset_id: str = "wikitext",
    dataset_name: str = "wikitext-103-raw-v1",
    validation_ratio: float = 0.0005,
    seq_len: int = 1024,
    preprocessing_num_proc: int = 64,
    stride: int = -1,
    ignore_train: bool = False,
    streaming: bool = True,
) -> datasets.DatasetDict:
    """Run basic tokenization and grouping to turn a Hugging Face Dataset (via `datasets`) into a torch.Dataset."""

    # Sanity check on input args
    stride = seq_len if stride < 0 else stride
    assert stride <= seq_len, f"Data grouping stride ({stride}) is smaller than sequence length: we are losing data."

    dataset = datasets.load_dataset(
        dataset_id, name=dataset_name, cache_dir=str(paths["dataset"]), keep_in_memory=not streaming,
        streaming=streaming,
    )

    # while HF datasets has a map method for both normal Dataset and streaming IterableDataset, the IterableDataset
    # version doesn't actually work with Trainer so we have to wrap it in this StreamedDataset nonsense.
    # The reasons are:
    # 1. IterableDataset transforms are pickled using pickle (not dill or something reasonable), so you have to be super
    # careful about what you pass to map. This is mostly Python's fault.
    # 2. IterableDataset doesn't have a __len__ method, which is required by Trainer for non-torch.utils.IterableDataset
    # types. You should be able to fix this by calling dataset.with_format(type="torch") but that's not working for
    # reason #1 (they use a local class which can't be pickled). You can work around this too (see ensure_format).
    # 3. IterableDataset, when successfully pickled, tries to `open` the urls of the dataset as files (rather than
    # downloading them), which of course explodes. I haven't figured out how to work around this yet.
    if streaming:
        detokenizer = DATASET_TOKENIZATION_REGISTRY.get(dataset_id, None)
        return IterableDatasetDict({
            k: StreamedDataset(dataset_id, dataset_name, k, str(paths["dataset"]), tokenizer, detokenizer, seq_len,
                               stride) for k, _ in dataset.items()
        })

    if "validation" not in dataset:
        assert "train" in dataset, "You must have train in dataset to make a validation dataset"
        # Create Dataset Split Cache Files
        train_fn, val_fn = [str(paths["dataset"] / dataset_id / f"{k}-split.hf") for k in ["train", "val"]]
        # TODO: what to do for streaming?
        dataset = dataset["train"].train_test_split(
            test_size=validation_ratio,
            train_indices_cache_file_name=train_fn,
            test_indices_cache_file_name=val_fn,
        )
        dataset["validation"] = dataset["test"]
        del dataset["test"]

    # Preprocess Dataset in a Streaming Fashion
    assert "train" in dataset, "Field `train` not in Dataset!"
    if ignore_train:
        del dataset["train"]
        assert len(dataset) > 0, "You can't set ignore_train = True when there is only train data"

    # First, Normalize Text if Necessary. Tokenization Strategies are in detokenization.py.
    dataset = auto_detokenize(dataset_id, dataset, paths["preprocessed"], preprocessing_num_proc, streaming)

    # Second, run straight-up tokenization
    if streaming:
        tokenized_dataset = dataset.map(
            _TokenizerFun(tokenizer),
            batched=True,
            remove_columns=list(next(iter(next(iter(dataset.values())))).keys()),
        )
    else:
        overwatch.info(f"Tokenizing Dataset via Multiprocessing with `{preprocessing_num_proc}` threads...")
        # Create Post-Tokenization Cache Paths
        post_tokenization_cache_files = {
            k: str(paths["preprocessed"] / dataset_id / "preprocessing" / "tokenization" / f"{k}-tokenized.hf")
            for k in dataset
        }
        # Create Parent Path of Cache Files
        (paths["preprocessed"] / dataset_id / "preprocessing" / "tokenization").mkdir(parents=True, exist_ok=True)

        def tokenize(examples: Dict[str, List[str]]) -> BatchEncoding:
            return tokenizer(examples["text"])

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            num_proc=preprocessing_num_proc,
            remove_columns=next(iter(dataset.values())).column_names,
            cache_file_names=post_tokenization_cache_files,
            load_from_cache_file=True,
        )

    # Finally, actually run chunking (collapse multiple sequences into a giant document to read `seq_len` chunks from)
    group = _GrouperFun(seq_len, stride)

    # From HF.Examples :: Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws
    # away a remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher
    # value might be slower to preprocess.
    #   - Sidd Note (3/11): We're dropping a max of 8M / 9B tokens... we're probably fine!
    if streaming:
        # .with_format(type="torch") doesn't work
        lm_dataset = _ensure_format(tokenized_dataset.map(
            group,
            batched=True,
        ))
    else:
        overwatch.info(f"Auto-Batching Dataset via Multiprocessing with `{preprocessing_num_proc}` threads...")
        # Create Post-Chunking Cache Paths
        post_chunking_cache_files = {
            k: str(paths["preprocessed"] / dataset_id / "preprocessing" / "chunking" / f"{k}-stride={stride}-chunked.hf")
            for k in dataset
        }
        # Create Parent Path of Cache Files
        (paths["preprocessed"] / dataset_id / "preprocessing" / "chunking").mkdir(parents=True, exist_ok=True)

        lm_dataset = tokenized_dataset.map(
            group,
            batched=True,
            num_proc=preprocessing_num_proc,
            cache_file_names=post_chunking_cache_files,
            load_from_cache_file=True,
        )

    return lm_dataset


def auto_detokenize(
        dataset_id: str, dataset: datasets.DatasetDict, preprocess_path: Path, preprocessing_num_proc: int = 4,
        streaming: bool = True,
) -> datasets.DatasetDict:
    if dataset_id in DATASET_TOKENIZATION_REGISTRY:
        overwatch.info(f"Detokenizing Dataset via Multiprocessing with `{preprocessing_num_proc}` threads...")

        if streaming:
            return dataset.map(DATASET_TOKENIZATION_REGISTRY[dataset_id])
        else:
            # Create Post-Detokenization Cache Paths
            post_detokenization_cache_files = {
                k: str(preprocess_path / dataset_id / "preprocessing" / "detokenization" / f"{k}-detokenized.hf")
                for k in dataset
            }

            # Create Parent Path of Cache Files
            (preprocess_path / dataset_id / "preprocessing" / "detokenization").mkdir(parents=True, exist_ok=True)

            return dataset.map(
                DATASET_TOKENIZATION_REGISTRY[dataset_id],
                num_proc=preprocessing_num_proc,
                cache_file_names=post_detokenization_cache_files,
                load_from_cache_file=True,
            )

    return dataset


def get_lambada(
    tokenizer: PreTrainedTokenizer,
    paths: Dict[str, Path],
    dataset_id: str = "lambada",
    dataset_name: str = None,
    validation_ratio: float = 0.0005,
    seq_len: int = 1024,
    preprocessing_num_proc: int = 4,
    stride: int = -1,
    ignore_train: bool = False,
) -> datasets.DatasetDict:
    """
    Run special tokenization and grouping for the Lambada dataset.

    Taken from https://github.com/NVIDIA/Megatron-LM/blob/main/tasks/zeroshot_gpt2/datasets.py
    """
    overwatch.info(f"Preprocessing LAMBADA Dataset via Multiprocessing with `{preprocessing_num_proc}` threads...")

    # Sanity check on Input Arguments
    stride = seq_len if stride < 0 else stride

    assert stride <= seq_len, f"Data grouping stride ({stride}) is smaller than sequence length; we are losing data."
    dataset = datasets.load_dataset(dataset_id, dataset_name, cache_dir=str(paths["dataset"]), keep_in_memory=True)
    del dataset["train"]

    def tokenize_and_group(example: Dict[str, str]) -> Dict[str, List[int]]:
        text = example["text"]
        last_token = text.split()[-1]
        start_idx = text.rfind(last_token)

        beginning_tokens, last_token = tokenizer.encode(text[:start_idx].strip()), tokenizer.encode(" " + last_token)
        num_pad = seq_len - len(beginning_tokens) - len(last_token)
        assert num_pad >= 0, "LAMBADA example is shorter than sequence length, will result in error."

        input_ids = beginning_tokens + last_token + [tokenizer.eos_token_id for _ in range(num_pad)]
        labels = [-100 for _ in beginning_tokens] + [tok for tok in last_token] + [-100 for _ in range(num_pad)]
        attention_mask = [1 for _ in range(len(beginning_tokens) + len(last_token))] + [0 for _ in range(num_pad)]

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    # Create Preprocessing Cache Paths
    post_preprocess_cache_files = {
        k: str(paths["preprocessed"] / "lambada" / "preprocessing" / f"{k}-processed.hf") for k in dataset
    }
    # Create Parent Path of Cache Files
    (paths["preprocessed"] / "lambada" / "preprocessing").mkdir(parents=True, exist_ok=True)

    processed_dataset = dataset.map(
        tokenize_and_group,
        batched=False,
        num_proc=preprocessing_num_proc,
        remove_columns=next(iter(dataset.values())).column_names,
        cache_file_names=post_preprocess_cache_files,
        load_from_cache_file=True,
    )

    return processed_dataset


# Mapping of eval dataset name -> HF ids, names, and method for generating dataset
ONLINE_EVAL_DATA_REGISTRY = {
    "wikitext": {"id": "wikitext", "name": "wikitext-103-raw-v1", "generator": get_auto_dataset},
    "lambada": {"id": "lambada", "name": None, "generator": get_lambada},
}


# more hax: there's a method in datasets called iterable_dataset that mixes in torch IterableDataset like this, but it
# creates an instance of a local class, and the local class isn't pickleable, so... we'll make our own mixed in version
class TorchIterableDataset(IterableDataset, torch.utils.data.IterableDataset):
    pass


def _ensure_format(dataset: IterableDatasetDict) -> IterableDatasetDict:
    return IterableDatasetDict({k: TorchIterableDataset(v._ex_iterable, v.info, v.split, "torch", v._shuffling) for k, v in dataset.items()})


# to make pickling happy
class _TokenizerFun:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: List[Dict[str, str]]):
        return self.tokenizer([ex["text"] for ex in examples])


class _GrouperFun:
    def __init__(self, seq_len: int, stride: int):
        self.seq_len = seq_len
        self.stride = stride

    def __call__(self, examples: Dict[str, Iterable[List[int]]]) -> Dict[str, List[List[int]]]:
        # Concatenate all the Texts
        concatenated: Dict[str, List[int]] = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])

        # Drop the "very last" bit of the dataset that doesn't fit into block size...
        total_length = ((total_length - self.seq_len + self.stride) // self.stride) * self.stride

        # Split by Chunks of Maximum Length
        result = {k: [t[i: i + self.seq_len] for i in range(0, total_length, self.stride)] for k, t in
                  concatenated.items()}
        result["labels"] = deepcopy(result["input_ids"])

        # Mask out losses in overlapping regions. If training data, string will be equal to seq_len
        for i, labels in enumerate(result["labels"]):
            if i == 0:
                continue
            for j in range(len(labels) - self.stride):
                labels[j] = -100
            result["labels"][i] = labels
        return result


class StreamedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_id: str, dataset_name: str, split: str, cache_dir: str, tokenizer: PreTrainedTokenizerBase,
                 detokenizer: Optional[Callable], seq_len: int, stride: int):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.detokenizer = detokenizer
        self.dataset = None

    def _tokenize(self, examples):
        tokenized = self.tokenizer(examples["text"])
        return tokenized

    def _init_dataset(self):
        dataset = datasets.load_dataset(self.dataset_id, self.dataset_name, split=self.split, cache_dir=self.cache_dir, streaming=True)
        if self.detokenizer:
            dataset = dataset.map(self.detokenizer, batched=False)

        tokenized = dataset.map(self._tokenize, batched=True, batch_size=1000, remove_columns=dataset.info.features.keys())
        grouped = tokenized.map(_GrouperFun(self.seq_len, self.stride), batched=True, batch_size=1000)
        self.dataset = grouped

    def __iter__(self):
        if self.dataset is None:
            self._init_dataset()

        return iter(self.dataset)



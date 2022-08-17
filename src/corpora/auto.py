"""
auto.py

Default Dataset/Corpus Utilities. Downloads (if necessary) from the Hugging Face `datasets` Hub, and organizes into
de-facto training, validation, and testing tests. Performs additional tokenization and normalization as well.
"""
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import datasets
from transformers import BatchEncoding, PreTrainedTokenizer

from src.corpora.detokenization import DATASET_TOKENIZATION_REGISTRY

from .indexer import IndexedDataset

# Nest Overwatch under root `mistral` logger, inheriting formatting!
from .tokenization_utils import batch_tokenize


overwatch = logging.getLogger("mistral.corpora.auto")


def build_indexed_dataset(
    tokenizer: PreTrainedTokenizer,
    paths: Dict[str, Path],
    dataset_id: str,
    dataset_name: Optional[str],
    dataset_dir: Optional[str],
    seq_len: int,
    stride: Optional[int] = None,
    preprocessing_num_proc: int = 64,
    ignore_train: bool = False,
    shuffle_seed: int = 42,
    train_shuffle_buffer_size: Optional[int] = 10000,
) -> Dict[str, IndexedDataset]:
    """Builds Indexed Datasets from a Dataset Dictionary."""

    dataset_key = dataset_id
    if dataset_name is not None:
        dataset_key = f"{dataset_name}-{dataset_id}"

    # First, Normalize Text if Necessary. Tokenization Strategies are in detokenization.py.
    if dataset_dir is not None:
        file_names = os.listdir(dataset_dir)
        file_type = os.path.splitext(file_names[0])[1][1:]
        dataset_files = {}
        dataset_files["train"] = [
            f"{dataset_dir}/{fn}" for fn in file_names if "train" in fn and fn.endswith(file_type)
        ]
        dataset_files["validation"] = [
            f"{dataset_dir}/{fn}" for fn in file_names if "validation" in fn and fn.endswith(file_type)
        ]
        file_type = "json" if file_type == "jsonl" else file_type
        assert file_type in ["json", "txt", "csv"]
        dataset = datasets.load_dataset(
            file_type,
            name=dataset_name,
            data_files=dataset_files,
            cache_dir=str(paths["dataset"]),
        )
    else:
        dataset = datasets.load_dataset(dataset_id, name=dataset_name, cache_dir=str(paths["dataset"]))

    if ignore_train and "train" in dataset:
        del dataset["train"]

    dataset = auto_detokenize(dataset_id, dataset, paths["preprocessed"], preprocessing_num_proc)

    # Create Post-Tokenization Cache Paths
    tokenization_cache = paths["preprocessed"] / dataset_key / "preprocessing" / "tokenization"
    tokenization_cache.mkdir(parents=True, exist_ok=True)

    post_tokenization_cache_files = {k: tokenization_cache / f"{k}-tokenized" for k in dataset}

    overwatch.info("Building Tokenized Indexed Dataset for {dataset_id}/{dataset_name}...")
    out_datasets = {}
    for k, ds in dataset.items():
        overwatch.info(f"Building Indexed Dataset for {k}")
        token_iter = batch_tokenize(ds, tokenizer, batch_size=1000)
        out_datasets[k] = IndexedDataset.build_or_load(token_iter, post_tokenization_cache_files[k], seq_len, stride)  # type: ignore

    if train_shuffle_buffer_size is not None and "train" in out_datasets:
        out_datasets["train"] = out_datasets["train"].seeded_shuffle(
            seed=shuffle_seed, buffer_size=train_shuffle_buffer_size
        )

    return out_datasets


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
) -> datasets.DatasetDict:
    """Run basic tokenization and grouping to turn a Hugging Face Dataset (via `datasets`) into a torch.Dataset."""

    # Sanity check on input args
    stride = seq_len if stride < 0 else stride
    assert stride <= seq_len, f"Data grouping stride ({stride}) is smaller than sequence length: we are losing data."
    dataset = datasets.load_dataset(dataset_id, name=dataset_name, cache_dir=str(paths["dataset"]))

    if "validation" not in dataset:
        assert "train" in dataset, "You must have train in dataset to make a validation dataset"
        # Create Dataset Split Cache Files
        train_fn, val_fn = [str(paths["dataset"] / dataset_id / f"{k}-split.hf") for k in ["train", "val"]]
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
    dataset = auto_detokenize(dataset_id, dataset, paths["preprocessed"], preprocessing_num_proc)

    # Second, run straight-up tokenization
    def tokenize(examples: Dict[str, List[str]]) -> BatchEncoding:
        return tokenizer(examples["text"])

    overwatch.info(f"Tokenizing Dataset via Multiprocessing with `{preprocessing_num_proc}` threads...")

    # Create Post-Tokenization Cache Paths
    post_tokenization_cache_files = {
        k: str(paths["preprocessed"] / dataset_id / "preprocessing" / "tokenization" / f"{k}-tokenized.hf")
        for k in dataset
    }
    # Create Parent Path of Cache Files
    (paths["preprocessed"] / dataset_id / "preprocessing" / "tokenization").mkdir(parents=True, exist_ok=True)

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        # tokenization is parallelized by huggingface's fast tokenizers
        num_proc=1 if tokenizer.is_fast else preprocessing_num_proc,
        remove_columns=next(iter(dataset.values())).column_names,
        cache_file_names=post_tokenization_cache_files,
        load_from_cache_file=True,
    )

    # Finally, actually run chunking (collapse multiple sequences into a giant document to read `seq_len` chunks from)
    def group(examples: Dict[str, Iterable[List[int]]]) -> Dict[str, List[List[int]]]:
        # Concatenate all the Texts
        concatenated: Dict[str, List[int]] = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])

        # Drop the "very last" bit of the dataset that doesn't fit into block size...
        total_length = ((total_length - seq_len + stride) // stride) * stride

        # Split by Chunks of Maximum Length
        result = {k: [t[i : i + seq_len] for i in range(0, total_length, stride)] for k, t in concatenated.items()}
        result["labels"] = deepcopy(result["input_ids"])

        # Mask out losses in overlapping regions. If training data, string will be equal to seq_len
        for i, labels in enumerate(result["labels"]):
            if i == 0:
                continue
            for j in range(len(labels) - stride):
                labels[j] = -100
            result["labels"][i] = labels
        return result

    # From HF.Examples :: Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws
    # away a remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher
    # value might be slower to preprocess.
    #   - Sidd Note (3/11): We're dropping a max of 8M / 9B tokens... we're probably fine!
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
    dataset_id: str, dataset: datasets.DatasetDict, preprocess_path: Path, preprocessing_num_proc: int = 4
) -> datasets.DatasetDict:
    if dataset_id in DATASET_TOKENIZATION_REGISTRY:
        overwatch.info(f"Detokenizing Dataset via Multiprocessing with `{preprocessing_num_proc}` threads...")

        # Create Post-Detokenization Cache Paths
        post_detokenization_cache_files = {
            k: str(preprocess_path / dataset_id / "preprocessing" / "detokenization" / f"{k}-detokenized.hf")
            for k in dataset
        }

        # Create Parent Path of Cache Files
        (preprocess_path / dataset_id / "preprocessing" / "detokenization").mkdir(parents=True, exist_ok=True)

        detokenized_dataset = dataset.map(
            DATASET_TOKENIZATION_REGISTRY[dataset_id],
            num_proc=preprocessing_num_proc,
            cache_file_names=post_detokenization_cache_files,
            load_from_cache_file=True,
        )
    else:
        detokenized_dataset = dataset

    return detokenized_dataset


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
        assert num_pad >= 0, "LAMBADA example is longer than sequence length, will result in error."

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

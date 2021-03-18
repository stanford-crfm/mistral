"""
auto.py

Default Dataset/Corpus Utilities. Downloads (if necessary) from the Hugging Face `datasets` Hub, and organizes into
de-facto training, validation, and testing tests. Performs additional tokenization and normalization as well.
"""
import logging
from pathlib import Path
from typing import Dict, List

import datasets
from transformers import BatchEncoding, PreTrainedTokenizer

# Nest Overwatch under root `mistral` logger, inheriting formatting!
from ..util.registry import DATASET_TOKENIZATION_STRATEGY


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
) -> datasets.Dataset:
    """ Run basic tokenization and grouping to turn a Hugging Face Dataset (via `datasets`) into a torch.Dataset. """
    # Sanity check on input args
    stride = seq_len if stride < 0 else stride
    assert stride <= seq_len, "Data grouping stride is smaller than sequence length: we are losing data."

    dataset = datasets.load_dataset(dataset_id, dataset_name, cache_dir=str(paths["dataset"]), keep_in_memory=True)

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

    # First, Normalize Text if Necessary. Tokenization Strategies are in dekoneizatoin.py.
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
        num_proc=preprocessing_num_proc,
        remove_columns=next(iter(dataset.values())).column_names,
        cache_file_names=post_tokenization_cache_files,
        load_from_cache_file=True,
    )

    # Finally, actually run chunking (collapse multiple sequences into a giant document to read `seq_len` chunks from)
    def group(examples: Dict[str, List[int]]) -> Dict[str, List[int]]:
        # Concatenate all the Texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])

        # Drop the "very last" bit of the dataset that doesn't fit into block size...
        total_length = ((total_length - seq_len + stride) // stride) * stride

        # Split by Chunks of Maximum Length
        result = {k: [t[i : i + seq_len] for i in range(0, total_length, stride)] for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        result["labels"][1:, :-stride] = -100
        return result

    # From HF.Examples :: Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws
    # away a remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher
    # value might be slower to preprocess.
    #   - Sidd Note (3/11): We're dropping a max of 8M / 9B tokens... we're probably fine!
    overwatch.info(f"Auto-Batching Dataset via Multiprocessing with `{preprocessing_num_proc}` threads...")

    # Create Post-Chunking Cache Paths
    post_chunking_cache_files = {
        k: str(paths["preprocessed"] / dataset_id / "preprocessing" / "chunking" / f"{k}-stride{stride}-chunked.hf")
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
    dataset_id: str, dataset: datasets.DatasetDict, preprocess_path: Path, preprocessing_num_proc: int = 8
) -> datasets.DatasetDict:
    if dataset_id in DATASET_TOKENIZATION_STRATEGY:
        overwatch.info(f"Detokenizing Dataset via Multiprocessing with `{preprocessing_num_proc}` threads...")
        # Create Post-Detokenization Cache Paths
        post_detokenization_cache_files = {
            k: str(preprocess_path / dataset_id / "preprocessing" / "detokenization" / f"{k}-detokenized.hf")
            for k in dataset
        }
        # Create Parent Path of Cache Files
        (preprocess_path / dataset_id / "preprocessing" / "detokenization").mkdir(parents=True, exist_ok=True)

        detokenized_dataset = dataset.map(
            DATASET_TOKENIZATION_STRATEGY[dataset_id],
            num_proc=preprocessing_num_proc,
            cache_file_names=post_detokenization_cache_files,
            load_from_cache_file=True,
        )
    else:
        detokenized_dataset = dataset
    return detokenized_dataset

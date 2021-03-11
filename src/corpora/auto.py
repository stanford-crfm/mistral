"""
auto.py

Default Dataset/Corpus Utilities. Downloads (if necessary) from the Hugging Face `datasets` Hub, and organizes into
de-facto training, validation, and testing tests. Performs additional tokenization and normalization as well.
"""
import logging
import os
from typing import Dict, List

import datasets
from quinine import Quinfig
from transformers import BatchEncoding, PreTrainedTokenizerBase


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.corpora.auto")


def get_auto_dataset(tokenizer: PreTrainedTokenizerBase, quinfig: Quinfig, paths: Dict[str, str]) -> datasets.Dataset:
    dataset = datasets.load_dataset(quinfig.dataset.id, cache_dir=paths["dataset"], keep_in_memory=True)

    if "validation" not in dataset:
        # Create Dataset Split Cache Files
        train_fn, val_fn = [
            os.path.join(paths["dataset"], quinfig.dataset.id, f"{k}-split.hf") for k in ["train", "val"]
        ]
        dataset = dataset["train"].train_test_split(
            test_size=quinfig.dataset.validation_ratio,
            train_indices_cache_file_name=train_fn,
            test_indices_cache_file_name=val_fn,
        )
        dataset["validation"] = dataset["test"]
        del dataset["test"]

    # Preprocess Dataset in a Streaming Fashion
    assert "train" in dataset, "Field `train` not in Dataset!"

    # First, run straight-up tokenization
    def tokenize(examples: Dict[str, List[str]]) -> BatchEncoding:
        return tokenizer(examples["text"])

    overwatch.info(f"Tokenizing Dataset via Multiprocessing with `{quinfig.dataset.num_proc}` threads...")

    # Create Post-Tokenization Cache Paths
    post_tokenization_cache_files = {
        k: os.path.join(paths["preprocessed"], "preprocessing", "tokenization", f"{k}-tokenized.hf") for k in dataset
    }
    os.makedirs(os.path.join(paths["preprocessed"], "preprocessing", "tokenization"), exist_ok=True)

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=quinfig.dataset.num_proc,
        remove_columns=dataset["train"].column_names,
        cache_file_names=post_tokenization_cache_files,
        load_from_cache_file=True,
    )

    # Second, actually run chunking (collapse multiple sequences into a giant document to read `seq_len` chunks from)
    def group(examples: Dict[str, List[int]]) -> Dict[str, List[int]]:
        # Concatenate all the Texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])

        # Drop the "very last" bit of the dataset that doesn't fit into block size...
        total_length = (total_length // quinfig.model.seq_len) * quinfig.model.seq_len

        # Split by chunks of Maximum Length - TODO 17 :: I don't like the fact that we precompute splits once...
        result = {
            k: [t[i : i + quinfig.model.seq_len] for i in range(0, total_length, quinfig.model.seq_len)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # From HF.Examples :: Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws
    # away a remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher
    # value might be slower to preprocess.
    #   - Sidd Note (3/11): We're dropping a max of 8M / 9B tokens... we're probably fine!
    overwatch.info(f"Auto-Batching Dataset via Multiprocessing with `{quinfig.dataset.num_proc}` threads...")

    # Create Post-Chunking Cache Paths
    post_chunking_cache_files = {
        k: os.path.join(paths["preprocessed"], "preprocessing", "chunking", f"{k}-chunked.hf") for k in dataset
    }
    os.makedirs(os.path.join(paths["preprocessed"], "preprocessing", "chunking"), exist_ok=True)

    lm_dataset = tokenized_dataset.map(
        group,
        batched=True,
        num_proc=quinfig.dataset.num_proc,
        cache_file_names=post_chunking_cache_files,
        load_from_cache_file=True,
    )

    return lm_dataset

"""
auto.py

Default Dataset/Corpus Utilities. Downloads (if necessary) from the Hugging Face `datasets` Hub, and organizes into
de-facto training, validation, and testing tests. Performs additional tokenization and normalization as well.
"""
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

import datasets
from transformers import BatchEncoding, PreTrainedTokenizer

from src.corpora.detokenization import DATASET_TOKENIZATION_REGISTRY


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.corpora.auto")


def get_auto_dataset(
    tokenizer: PreTrainedTokenizer,
    paths: Dict[str, Path],
    dataset_id: str = "wikitext",
    dataset_name: str = "wikitext-103-raw-v1",
    dataset_source: str = "hub",
    dataset_ratios: str = None,
    data_dir: str = None,
    seed: int = 21,
    validation_ratio: float = 0.0005,
    seq_len: int = 1024,
    preprocessing_num_proc: int = 64,
    stride: int = -1,
    ignore_train: bool = False,
    detokenize: bool = True,
) -> datasets.DatasetDict:
    """Run basic tokenization and grouping to turn a Hugging Face Dataset (via `datasets`) into a torch.Dataset."""

    # Sanity check on input args
    stride = seq_len if stride < 0 else stride
    assert stride <= seq_len, f"Data grouping stride ({stride}) is smaller than sequence length: we are losing data."

    # Load initial datasets
    ds_id_list, ds_name_list, ds_source_list, ds_dir_list = (
        dataset_id.split(","),
        dataset_name.split(","),
        dataset_source.split(","),
        data_dir.split(","),
    )
    init_datasets = {"train": [], "validation": []}
    for (ds_id, ds_name, ds_source, ds_dir) in zip(ds_id_list, ds_name_list, ds_source_list, ds_dir_list):
        if ds_source == "hub" or ds_source is None:
            dataset = datasets.load_dataset(ds_id, name=ds_name, data_dir=ds_dir, cache_dir=str(paths["dataset"]))
        elif os.path.isdir(ds_source):
            file_names = os.listdir(ds_source)
            file_type = os.path.splitext(file_names[0])[1][1:]
            file_type = "json" if file_type == "jsonl" else file_type
            ds_files = {
                "train": [f"{ds_source}/{fn}" for fn in file_names if "train" in fn],
                "validation": [f"{ds_source}/{fn}" for fn in file_names if "validation" in fn],
            }
            dataset = datasets.load_dataset(
                file_type,
                name=ds_name,
                data_files=ds_files,
                cache_dir=str(paths["dataset"]),
            )
        if "validation" not in dataset:
            assert "train" in dataset, "You must have train in dataset to make a validation dataset"
            # Create Dataset Split Cache Files
            train_fn, val_fn = [str(paths["dataset"] / dataset_id / f"{k}-split.hf") for k in ["train", "validation"]]
            dataset = dataset["train"].train_test_split(
                test_size=validation_ratio,
                train_indices_cache_file_name=train_fn,
                test_indices_cache_file_name=val_fn,
            )
            dataset["validation"] = dataset["test"]
            del dataset["test"]

        init_datasets["train"].append(dataset["train"])
        init_datasets["validation"].append(dataset["validation"])
    
    # Interleave datasets
    dataset_ratios = [float(r) for r in dataset_ratios.split(",")] if dataset_ratios is not None else dataset_ratios
    dataset = datasets.DatasetDict()
    for split in ["train", "validation"]:
        dataset[split] = datasets.combine.interleave_datasets(datasets=init_datasets[split], probabilities=dataset_ratios, seed=seed)

    # Preprocess Dataset in a Streaming Fashion
    assert "train" in dataset, "Field `train` not in Dataset!"
    if ignore_train:
        del dataset["train"]
        assert len(dataset) > 0, "You can't set ignore_train = True when there is only train data"

    # First, Normalize Text if Necessary. Tokenization Strategies are in detokenization.py.
    if detokenize:
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

"""
detokenization.py

Handle detokenization for different dataset for zero-shot LM evaluation.
"""
import logging
import datasets
import re
from pathlib import Path
from typing import Dict

# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.corpora.auto")


def auto_detokenize(
    dataset_id: str, dataset: datasets.DatasetDict, preprocess_path: Path, preprocessing_num_proc: int = 8
) -> datasets.DatasetDict:
    if dataset_id in detokenizer_dict:
        overwatch.info(f"Detokenizing Dataset via Multiprocessing with `{preprocessing_num_proc}` threads...")
        # Create Post-Detokenization Cache Paths
        post_detokenization_cache_files = {
            k: str(preprocess_path / dataset_id / "preprocessing" / "detokenization" / f"{k}-detokenized.hf")
            for k in dataset
        }
        # Create Parent Path of Cache Files
        (preprocess_path / dataset_id / "preprocessing" / "detokenization").mkdir(parents=True, exist_ok=True)

        detokenized_dataset = dataset.map(
            detokenizer_dict[dataset_id],
            num_proc=preprocessing_num_proc,
            cache_file_names=post_detokenization_cache_files,
            load_from_cache_file=True,
        )
    else:
        detokenized_dataset = dataset
    return detokenized_dataset


def wikitext_detokenize(example: Dict[str, str]) -> Dict[str, str]:
    """
    Wikitext is whitespace tokenized and we remove these whitespaces.
    Taken from https://github.com/NVIDIA/Megatron-LM/blob/main/tasks/zeroshot_gpt2/detokenizer.py
    """
    # contractions
    text = example["text"]
    text = text.replace("s '", "s'")
    text = re.sub(r"/' [0-9]/", r"/'[0-9]/", text)
    # number separators
    text = text.replace(" @-@ ", "-")
    text = text.replace(" @,@ ", ",")
    text = text.replace(" @.@ ", ".")
    # punctuation
    text = text.replace(" : ", ": ")
    text = text.replace(" ; ", "; ")
    text = text.replace(" . ", ". ")
    text = text.replace(" ! ", "! ")
    text = text.replace(" ? ", "? ")
    text = text.replace(" , ", ", ")
    # double brackets
    text = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", text)
    text = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", text)
    text = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", text)
    text = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', text)
    text = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", text)
    # miscellaneous
    text = text.replace("= = = =", "====")
    text = text.replace("= = =", "===")
    text = text.replace("= =", "==")
    text = text.replace(" " + chr(176) + " ", chr(176))
    text = text.replace(" \n", "\n")
    text = text.replace("\n ", "\n")
    text = text.replace(" N ", " 1 ")
    text = text.replace(" 's", "'s")

    return {"text": text}


detokenizer_dict = {"wikitext": wikitext_detokenize}

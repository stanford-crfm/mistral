"""
detokenization.py

Handle detokenization for different dataset for zero-shot LM evaluation.
"""
import logging
import re
from typing import Dict


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.corpora.auto")


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


DATASET_TOKENIZATION_STRATEGY = {"wikitext": wikitext_detokenize}

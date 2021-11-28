"""
auto_mlm.py

Default Dataset/Corpus Utilities. Downloads (if necessary) from the Hugging Face `datasets` Hub, and organizes into
de-facto training, validation, and testing tests. Performs additional tokenization and normalization as well.
"""
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List

import datasets
from transformers import BatchEncoding, PreTrainedTokenizer

from src.corpora.detokenization import DATASET_TOKENIZATION_REGISTRY


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.corpora.auto_mlm")


def get_auto_mlm_dataset(
    tokenizer: PreTrainedTokenizer,
    paths: Dict[str, Path],
    dataset_id: str = "wikitext",
    dataset_name: str = "wikitext-103-raw-v1",
    validation_ratio: float = 0.0005,
    seq_len: int = 1024,
    preprocessing_num_proc: int = 64,
    stride: int = -1,
    ignore_train: bool = False,
    mode: str = "nsp",
    random_seed: int = 12345,
) -> datasets.DatasetDict:
    """Run basic tokenization and grouping to turn a Hugging Face Dataset (via `datasets`) into a torch.Dataset."""

    # Sanity check on input args
    stride = seq_len if stride < 0 else stride
    assert stride <= seq_len, f"Data grouping stride ({stride}) is smaller than sequence length: we are losing data."
    dataset = datasets.load_dataset(
        dataset_id, name=dataset_name, cache_dir=str(paths["dataset"]), keep_in_memory=True
    )

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

    # DEBUG
    # del dataset["train"]

    # First, Normalize Text if Necessary. Tokenization Strategies are in detokenization.py.
    dataset = auto_detokenize(dataset_id, dataset, paths["preprocessed"], preprocessing_num_proc)

    def process_1seq():
        # Second, run straight-up tokenization
        def tokenize(examples: Dict[str, List[str]]) -> BatchEncoding:
            return tokenizer(examples["text"], add_special_tokens=False)

        overwatch.info(f"Tokenizing Dataset via Multiprocessing with `{preprocessing_num_proc}` threads...")

        # Create Post-Tokenization Cache Paths
        post_tokenization_cache_files = {
            k: str(paths["preprocessed"] / dataset_id / "preprocessing-1seq" / "tokenization" / f"{k}-tokenized.hf")
            for k in dataset
        }
        # Create Parent Path of Cache Files
        (paths["preprocessed"] / dataset_id / "preprocessing-1seq" / "tokenization").mkdir(parents=True, exist_ok=True)

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

            # To accomodate [CLS] [SEP]
            _seq_len = seq_len - 2
            _stride = stride - 2

            # Drop the "very last" bit of the dataset that doesn't fit into block size...
            total_length = ((total_length - _seq_len + _stride) // _stride) * _stride

            # Split by Chunks of Maximum Length
            cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
            result = {}
            result["input_ids"] = [
                [cls_id] + concatenated["input_ids"][i : i + _seq_len] + [sep_id]
                for i in range(0, total_length, _stride)
            ]
            result["attention_mask"] = [
                [1] + concatenated["attention_mask"][i : i + _seq_len] + [1] for i in range(0, total_length, _stride)
            ]
            result["token_type_ids"] = [
                [0] + concatenated["token_type_ids"][i : i + _seq_len] + [0] for i in range(0, total_length, _stride)
            ]
            return result

        # From HF.Examples :: Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws
        # away a remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher
        # value might be slower to preprocess.
        #   - Sidd Note (3/11): We're dropping a max of 8M / 9B tokens... we're probably fine!
        overwatch.info(f"Auto-Batching Dataset via Multiprocessing with `{preprocessing_num_proc}` threads...")

        # Create Post-Chunking Cache Paths
        post_chunking_cache_files = {
            k: str(
                paths["preprocessed"]
                / dataset_id
                / "preprocessing-1seq"
                / "chunking"
                / f"{k}-stride={stride}-chunked.hf"
            )
            for k in dataset
        }
        # Create Parent Path of Cache Files
        (paths["preprocessed"] / dataset_id / "preprocessing-1seq" / "chunking").mkdir(parents=True, exist_ok=True)

        lm_dataset = tokenized_dataset.map(
            group,
            batched=True,
            num_proc=preprocessing_num_proc,
            cache_file_names=post_chunking_cache_files,
            load_from_cache_file=True,
        )

        return lm_dataset

    def process_nsp():
        rng = random.Random(random_seed)

        def sentence_split(text):
            text = text.strip()
            sents = [s.strip() for s in text.split(". ")]
            sents = [s + "." for s in sents[:-1]] + sents[-1:]
            sents = [s for s in sents if s]
            return sents

        def get_docs(examples):
            num_sents_per_doc = []
            sents_all = []
            for doc in examples["text"]:
                doc = doc.strip()
                if len(doc) > 0:
                    sents = sentence_split(doc)
                    if len(sents) > 0:
                        sents_all += sents
                        num_sents_per_doc.append(len(sents))
            assert len(sents_all) == sum(num_sents_per_doc)
            sents_all_toked = tokenizer(sents_all, add_special_tokens=False)["input_ids"]
            docs = []
            sid = 0
            for num_sents in num_sents_per_doc:
                doc = [s for s in sents_all_toked[sid : sid + num_sents] if len(s) > 0]
                sid += num_sents
                if len(sum(doc, [])) >= 32:
                    docs.append(doc)
            assert sid == len(sents_all)
            return docs

        def get_docs_for_HFbookcorpus(examples):
            # HuggingFace BookCorpus is not structured as documents; it is a list of sentences.
            # So we randomly chunk them into "documents", e.g. `doc_len` (=100) sentences as one document
            sents = []
            for sent in examples["text"]:
                sent = sent.strip()
                if sent:
                    sents.append(sent)
            sents_all_toked = tokenizer(sents, add_special_tokens=False)["input_ids"]
            doc_len = 100
            docs = []
            for i in range(0, len(sents_all_toked), doc_len):
                doc = [s for s in sents_all_toked[i : i + doc_len] if len(s) > 0]
                if len(sum(doc, [])) >= 32:
                    docs.append(doc)
            return docs

        def tokenize_function(examples):
            # For NextSentencePrediction (NSP), we need to prepare documents where each entry is a list of sentences: [[doc1sent1, doc1sent2], [doc2sent1, doc2sent2]]
            if dataset_id == "bookcorpus":
                all_documents = get_docs_for_HFbookcorpus(examples)
            elif dataset_id in ["wikipedia", "wikitext"]:
                all_documents = get_docs(examples)
            else:
                raise NotImplementedError

            instances: dict = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "next_sentence_label": []}
            for document_index in range(len(all_documents)):
                _instances = create_nsp_instances(
                    rng, tokenizer, all_documents, document_index, max_seq_length=seq_len
                )
                for k in instances:
                    instances[k] += _instances[k]
            return instances

        post_tokenization_cache_files = {
            k: str(paths["preprocessed"] / dataset_id / "preprocessing-nsp" / "tokenization" / f"{k}-tokenized.hf")
            for k in dataset
        }
        # Create Parent Path of Cache Files
        (paths["preprocessed"] / dataset_id / "preprocessing-nsp" / "tokenization").mkdir(parents=True, exist_ok=True)

        overwatch.info(f"Tokenizing and Creating NSP instances with `{preprocessing_num_proc}` threads...")
        lm_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_proc,
            remove_columns=next(iter(dataset.values())).column_names,
            cache_file_names=post_tokenization_cache_files,
            load_from_cache_file=True,
        )
        return lm_dataset

    if mode == "nsp":
        lm_dataset = process_nsp()
    else:
        lm_dataset = process_1seq()

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


def create_nsp_instances(rng, tokenizer, all_documents, document_index, max_seq_length, short_seq_prob=0.1):
    """
    Creates NextSentencePrediction instances for a single document.
    Taken from https://github.com/google-research/bert/blob/master/create_pretraining_data.py
    """
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances: dict = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "next_sentence_label": []}
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1, (len(tokens_a), len(tokens_b), "is_random_next", is_random_next)

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append(cls_id)
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append(sep_id)
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append(sep_id)
                segment_ids.append(1)

                assert len(tokens) == len(segment_ids)
                instances["input_ids"].append(tokens)
                instances["attention_mask"].append([1] * len(tokens))
                instances["token_type_ids"].append(segment_ids)
                instances["next_sentence_label"].append(int(is_random_next))
            current_chunk = []
            current_length = 0
        i += 1
    return instances


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """
    Truncates a pair of sequences to a maximum sequence length.
    Taken from https://github.com/google-research/bert/blob/master/create_pretraining_data.py
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


# Mapping of eval dataset name -> HF ids, names, and method for generating dataset
ONLINE_EVAL_DATA_REGISTRY_MLM = {
    "wikitext": {"id": "wikitext", "name": "wikitext-103-raw-v1", "generator": get_auto_mlm_dataset},
    "wikipedia": {"id": "wikipedia", "name": "20200501.en", "generator": get_auto_mlm_dataset},
    "bookcorpus": {"id": "wikipedia", "name": "plain_text", "generator": get_auto_mlm_dataset},
}

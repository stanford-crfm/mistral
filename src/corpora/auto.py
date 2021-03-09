"""
auto.py

Default Dataset/Corpus Utilities. Downloads (if necessary) from the Hugging Face `datasets` Hub, and organizes into
de-facto training, validation, and testing tests. Performs additional tokenization and normalization as well.
"""
import datasets
from typing import Dict, List

__all__ = ["get_dataset"]


def get_dataset(tokenizer, quinfig, paths, overwatch):
    dataset = datasets.load_dataset(quinfig.dataset.id, quinfig.dataset.name, cache_dir=paths["dataset"])

    # TODO 7 -- For Text Corpora that DO NOT have pre-defined validation sets -- we need to create our own.
    #   Reference: https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py#L214
    if "validation" not in dataset:
        err = "Automatic Creation of Validation Dataset is not yet implemented!"
        overwatch.error(err)
        raise NotImplementedError(err)

    # Preprocess Dataset in a Streaming Fashion --> TODO 14 :: Validate that this Assertion always holds
    assert "train" in dataset

    # TODO -2 :: wrap data prep in separate function / file for cleanliness
    # First, run straight-up tokenization
    def tokenize(examples: Dict[str, List[int]]) -> Dict[str, List[int]]:
        return tokenizer(examples["text"])

    overwatch.info(f"Tokenizing Dataset via Multiprocessing with `{quinfig.dataset.num_proc}` threads...")
    # TODO -1 (Laurel's counting backwards) :: Check reloading with HF caches. If we save trainer.py, will it trigger the cache to be stale?

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=quinfig.dataset.num_proc,
        remove_columns=dataset["train"].column_names,  # TODO 15 :: This line may save marginally on memory?
        load_from_cache_file=True,
    )

    # Second, actually run chunking (collapse multiple sequences into a giant document to read `seq_len` chunks from)
    def group(examples: Dict[str, List[int]]) -> Dict[str, List[int]]:
        # Concatenate all the Texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])

        # Drop the "very last" bit of the dataset that doesn't fit into block size...
        # TODO 16 :: If someone really, really feels like it they can implement the wraparound logic...
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
    # TODO 18 :: Fix this so it's cleaner - I don't like dropping text, and this code (single split) is bad if we're
    #   running multiple epochs of training... To be honest, can probably go back to just the `Tempest` dataset class!
    overwatch.info(f"Auto-Batching Dataset via Multiprocessing with `{quinfig.dataset.num_proc}` threads...")
    lm_dataset = tokenized_dataset.map(
        group,
        batched=True,
        batch_size=1000,  # Default value in HF --> should probably tweak this as part of 17?
        num_proc=quinfig.dataset.num_proc,
        load_from_cache_file=True,  # TODO 34 :: For some reason, we never seem to be using the cache? Fix!
    )
    return lm_dataset

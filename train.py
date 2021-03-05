"""
train.py

Run large-scale language model training across various datasets and model types, leveraging Hydra and
`torch.distributed.launch` to support multiple models, datasets, and training configurations. Includes code for
loading, preprocessing, and building `torch.Datasets` from given dataset identifier (via huggingface.Datasets),
initializing models of varying architectures, and training.

Supported Models:
    - GPT-2

Supported Datasets:
    - WikiText-103
    - OpenWebText [WIP]

Provides additional scripting for logging, interfacing with Weights & Biases, and serializing/saving model checkpoints.

Reference:
    - https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py

=>> A Project Mercury Endeavor
"""
import math
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from quinine import QuinineArgumentParser
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src.overwatch import get_overwatch
from src.util import REGISTRY, create_paths


def train() -> None:
    # Parse Quinfig (via Quinine Argparse Binding -- TODO 3 :: Create Quinfig Schema && Fix extra \n in Quinfig!)
    print("[*] Mercury :: Launching =>>> \N{rocket} \N{see-no-evil monkey} \N{rocket}", end="")
    quinfig = QuinineArgumentParser().parse_quinfig()
    print('\t=>> "This wind, it is not an ending..." (Robert Jordan - A Memory of Light)\n')

    # Create Unique Run Name (for Logging, Checkpointing, and W&B) :: Initialize all Directories
    run_id = quinfig.run_id
    # TODO -5 :: Add a custom run_name or something to the path below so it's not just run_id or the default
    if run_id is None:
        # TODO 4 :: Fix Quinfig (@Karan) so that nested inheritance doesn't require "strings"
        run_id = (
            f"{quinfig.model['id']}-d={quinfig.dataset['id']}-n={quinfig.infra.nodes}-g={quinfig.infra.gpus}+"
            f"{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
        )
    paths = create_paths(run_id, quinfig.model["id"], quinfig.artifacts.run_dir, quinfig.artifacts.cache_dir)

    # Overwatch :: Setup & Configure Console/File Logger --> Handle Process 0 vs. other Process Logging!
    overwatch = get_overwatch(os.path.join(paths["runs"], f"{run_id}.log"), quinfig.log_level, rank=quinfig.infra.rank)
    overwatch.info(f"Starting Run: {run_id}...")

    # Set Randomness
    overwatch.info(f"Setting Random Seed to {quinfig.seed}!")
    random.seed(quinfig.seed)
    np.random.seed(quinfig.seed)
    torch.manual_seed(quinfig.seed)

    # TODO 5 -- @Karan :: Quinfig should support overriding top-level config arguments from CLI `--resume True`!
    # TODO 6 -- Resume from Checkpoint Behavior!
    #   See: https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py#L166
    if quinfig.resume:
        err = "Resume behavior is not yet implemented!"
        overwatch.error(err)
        raise NotImplementedError(err)

    # Create Tokenizer
    overwatch.info(f"Fetching Hugging Face [Fast] AutoTokenizer for Model: `{REGISTRY[quinfig.model['id']]}`...")
    if quinfig.model["pretrained_tokenizer"]:
        tokenizer = AutoTokenizer.from_pretrained(REGISTRY[quinfig.model["id"]], cache_dir=paths["tokenizer"])
    else:
        overwatch.error("Tokenizer Training/Initialization (from Scratch) not yet implemented!")
        raise NotImplementedError()

    # Load Dataset w/ Preprocessing, Batching, and Collating
    # TODO 25 :: Make Dataset Creation & Processing Modular + Clean --> Relegate to `src.corpora.auto`
    overwatch.info(f"Downloading and Preprocessing Dataset `{quinfig.dataset['id']}`...")
    dataset = load_dataset(quinfig.dataset["id"], quinfig.dataset["name"], cache_dir=paths["dataset"])

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

    overwatch.info(f"Tokenizing Dataset via Multiprocessing with `{quinfig.dataset['num_proc']}` threads...")
    # TODO -1 (Laurel's counting backwards) :: Check reloading with HF caches. If we save trainer.py, will it trigger the cache to be stale?
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=quinfig.dataset["num_proc"],
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
        total_length = (total_length // quinfig.model["seq_len"]) * quinfig.model["seq_len"]

        # Split by chunks of Maximum Length - TODO 17 :: I don't like the fact that we precompute splits once...
        result = {
            k: [t[i : i + quinfig.model["seq_len"]] for i in range(0, total_length, quinfig.model["seq_len"])]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # From HF.Examples :: Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws
    # away a remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher
    # value might be slower to preprocess.
    # TODO 18 :: Fix this so it's cleaner - I don't like dropping text, and this code (single split) is bad if we're
    #   running multiple epochs of training... To be honest, can probably go back to just the `Tempest` dataset class!
    overwatch.info(f"Auto-Batching Dataset via Multiprocessing with `{quinfig.dataset['num_proc']}` threads...")
    lm_dataset = tokenized_dataset.map(
        group,
        batched=True,
        batch_size=1000,  # Default value in HF --> should probably tweak this as part of 17?
        num_proc=quinfig.dataset["num_proc"],
        load_from_cache_file=True,  # TODO 34 :: For some reason, we never seem to be using the cache? Fix!
    )

    # Create Model Configuration
    # TODO 26 :: Make Model Creation & Processing Modular + Clean --> Relegate to `src.models.auto`
    overwatch.info(f"Fetching Hugging Face AutoConfig for Model: `{REGISTRY[quinfig.model['id']]}`...")
    model_config = AutoConfig.from_pretrained(REGISTRY[quinfig.model["id"]], cache_dir=paths["configs"])

    # Initialize Model
    # TODO 27 :: Make sure weight initialization follows GPT-2 Paper & Best Practices [it does not currently]
    overwatch.info(f"Initializing Tabula Rasa Model from Configuration: `{REGISTRY[quinfig.model['id']]}`...")
    model = AutoModelForCausalLM.from_config(model_config)
    model.resize_token_embeddings(len(tokenizer))

    # Initialize Training Arguments from Quinfig
    # TODO 20 :: Clean this up in a neat way -- probably overwrite in grand-child config itself... but path injection?
    training_args = quinfig.training_arguments
    training_args["run_name"] = run_id
    training_args["output_dir"] = paths["runs"]
    training_args["logging_dir"] = paths["logs"]
    training_args["seed"] = quinfig.seed
    training_args["local_rank"] = quinfig.infra["rank"]
    training_args = TrainingArguments(**quinfig.training_arguments)

    # Important - Note that by default if multiple GPUs available on node, HF.Trainer defaults to `torch.DataParallel`
    #   which is almost always worse in efficiency than the DDP equivalent. So basically, always run with DDP!
    # TODO 21 :: Set up DDP (Single-Node), DDP (Multi-Node) Training + Mixed Precision Training
    # TODO 22 :: Setup DeepSpeed Training
    # TODO 23 :: Setup FairScale Training
    # TODO 24 :: Figure out best combination of DeepSpeed & FairScale (if they even can be combined well)

    # Initialize Trainer, with the relevant arguments
    # TODO 29 :: Setup W&B using Environment Variables (Pass to Trainer)
    # TODO 30 :: Setup Custom Logger (File/JSON Logger from `Tempest) Callback and add here!
    # TODO 31 :: Add Environment/Climate Tracker from `Tempest`/Peter Henderson here as well
    # TODO 32 :: Make sure we're using the right opt/schedule... should be configured by `training_args` so check!
    # TODO 33 :: Pass in `compute_metrics` for correct evaluation metrics --> Perplexity! Do during train as well?
    overwatch.info("Initializing Model Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,  # De Facto Collator uses Padding, which we DO NOT want!
    )

    # Training Time!
    # TODO 6 -- Resume from Checkpoint Behavior!
    #   See: https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py#L369
    overwatch.info("Training...")
    train_result = trainer.train()
    trainer.save_model()

    # Get and Log Metrics --> TODO 28 :: Is this necessary? Separately - we should write a Custom Simplified Logger!
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()  # No idea what this does...

    # Evaluation Time
    overwatch.info("Evaluating...")
    eval_result = trainer.evaluate()

    # Compute PPL and Log
    perplexity = math.exp(eval_result["eval_loss"])
    results = {"perplexity": perplexity}
    trainer.log_metrics("eval", results)
    trainer.save_metrics("eval", results)

    overwatch.info("...and that's all folks!")


if __name__ == "__main__":
    train()

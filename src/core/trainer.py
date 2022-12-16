"""
trainer.py

Custom Hugging Face Trainer that allows for online eval of multiple datasets.
"""
import collections
import logging
import time
from dataclasses import dataclass  # type: ignore
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers.trainer_pt_utils import IterableDatasetShard


try:
    from torchdata.datapipes.iter import IterDataPipe
except ImportError:
    from torch.utils.data import IterDataPipe

from transformers import (
    AutoModelForCausalLM,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollator
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, speed_metrics


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.core.trainer")


class OnlineBenchmarkTrainer(Trainer):
    """
    Trainer that handles online evaluation datasets -- e.g., LAMBADA and Wikitext 103 Perplexity Scores.

    Overrides `evaluate` to trigger eval on each online dataset.
    """

    control: Any
    _globalstep_last_logged: int

    def __init__(
        self,
        model: AutoModelForCausalLM,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        dataset_name: str = "unknown",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        custom_eval_datasets: Optional[Dict[str, Dataset]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super(OnlineBenchmarkTrainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.dataset_name = dataset_name
        custom_eval_datasets = custom_eval_datasets if custom_eval_datasets is not None else {}

        # No idea why, but you can't use a dict to store the datasets. They must be stored separately as class objects.
        # It might be related to how module need custom ModuleDicts for dictionaries to work with distributed models.
        self.wikitext_dataset = custom_eval_datasets.get("wikitext", None)
        self.lambada_dataset = custom_eval_datasets.get("lambada", None)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_ppl_datasets: bool = True,
    ) -> Dict[str, float]:

        # Normal Evaluate -- this calls the on_evaluate callback
        metrics = super(OnlineBenchmarkTrainer, self).evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Create New Metrics Dictionary --> TODO trainer.A :: Fix so doesn't explicitly assume OpenWebText
        metrics = {
            f"eval_{self.dataset_name}_loss": metrics["eval_loss"],
            f"eval_{self.dataset_name}_ppl": np.exp(metrics["eval_loss"]),
            f"eval_{self.dataset_name}_runtime": metrics["eval_runtime"],
            f"eval_{self.dataset_name}_samples_per_second": metrics["eval_samples_per_second"],
            "epoch": metrics.get("epoch"),
        }
        self.log(metrics)
        if not eval_ppl_datasets:
            return metrics

        # Start Memory Tracker
        self._memory_tracker.start()

        # Iterate over each Online Evaluation Dataset - Store New Metrics for Control Call
        new_dataset_metrics = {}
        if self.wikitext_dataset is not None:
            output_metrics = self.single_dataset_eval("wikitext", self.wikitext_dataset, metric_key_prefix)
            new_dataset_metrics.update(output_metrics)
            self.log(output_metrics)

        if self.lambada_dataset is not None:
            output_metrics = self.single_dataset_eval("lambada", self.lambada_dataset, metric_key_prefix)
            new_dataset_metrics.update(output_metrics)
            self.log(output_metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, new_dataset_metrics)
        self._memory_tracker.stop_and_update_metrics(new_dataset_metrics)
        metrics.update(new_dataset_metrics)

        return metrics

    def single_dataset_eval(self, dataset_name: str, dataset: Dataset, metric_key_prefix: str) -> Dict[str, float]:
        """Run Perplexity Evaluation on a Single Dataset."""
        custom_metric_key_prefix = f"{metric_key_prefix}_{dataset_name}"
        if dataset is not None and not isinstance(dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(dataset)
        start_time = time.time()
        output = self.prediction_loop(
            eval_dataloader,
            description=f"Evaluation {dataset_name}",
            prediction_loss_only=True,
            metric_key_prefix=custom_metric_key_prefix,
        )
        n_samples = len(dataset) if dataset is not None else 1
        output.metrics.update(speed_metrics(custom_metric_key_prefix, start_time, n_samples))

        # Compute perplexity --- Note :: this is unadjusted
        ppl = np.exp(output.metrics[f"{custom_metric_key_prefix}_loss"])
        output.metrics.update({f"{custom_metric_key_prefix}_ppl": ppl})

        return output.metrics

    def get_train_dataloader(self) -> DataLoader:
        """ensures we're shuffling if we're using a new-style (iterable) dataset"""
        if isinstance(self.train_dataset, IterDataPipe):
            train_dataset = DataLoader(
                self.train_dataset,
                shuffle=True,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return train_dataset
        else:
            return super().get_train_dataloader()


@dataclass
class LMDataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, examples: List[BatchEncoding]):
        batch = BatchEncoding(data={k: torch.tensor([v[k] for v in examples]) for k in examples[0].keys()})

        if "labels" in batch:
            labels = batch["labels"]
        else:
            labels = batch["input_ids"]

        if self.tokenizer.pad_token_id is not None:
            labels = labels.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch

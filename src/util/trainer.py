"""
trainer.py

Customer HF Trainer that allows for online eval of multiple datasets.
"""
import collections
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import EvalPrediction, speed_metrics


overwatch = logging.getLogger("mistral.util.trainer")


class OnlineBenchmarkTrainer(Trainer):
    """
    Trainer that handles online evaluation datasets -- e.g., LAMBADA and Wikitext 103 Perplexity Scores.

    Overrides `evaluate` to trigger eval on each online dataset.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
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
        custom_eval_datasets = custom_eval_datasets if custom_eval_datasets is not None else {}

        # No idea why, but you can't use a dict to store the datasets. They must be stored separately as class objects.
        # It might be related to how module need custom ModuleDicts for dictionaries to work with distributed models.
        #   =>> But who knows?
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
        if not eval_ppl_datasets:
            return metrics
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
        """ Run Perplexity Evaluation on a Single Dataset. """
        custom_metric_key_prefix = f"{dataset_name}-{metric_key_prefix}"
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

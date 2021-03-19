"""
trainer.py

Customer HF Trainer that allows for online eval of multiple datasets.
"""
import collections
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction, speed_metrics


overwatch = logging.getLogger("mistral.util.trainer")


class OnlineBenchmarkTrainer(Trainer):
    """
    Trainer that handles online evaluation datasets -- e.g., LAMBADA and Wikitext 103 Perplexity Scores.

    Overrides `evaluate` to trigger eval on each online dataset.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
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
        self.custom_eval_datasets = custom_eval_datasets if custom_eval_datasets is not None else {}

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

        # Iterate over each online eval dataset
        for custom_dataset_name, custom_eval_dataset in self.custom_eval_datasets.items():
            if custom_eval_dataset is not None and not isinstance(custom_eval_dataset, collections.abc.Sized):
                raise ValueError("eval_dataset must implement __len__")
            custom_metric_key_prefix = f"{metric_key_prefix}_{custom_dataset_name}"
            eval_dataloader = self.get_eval_dataloader(custom_eval_dataset)
            start_time = time.time()
            output = self.prediction_loop(
                eval_dataloader,
                description=f"Evaluation {custom_dataset_name}",
                prediction_loss_only=True,
                metric_key_prefix=custom_metric_key_prefix,
            )
            n_samples = len(custom_eval_dataset)
            output.metrics.update(speed_metrics(custom_metric_key_prefix, start_time, n_samples))
            # Compute perplexity --- Note :: this is unadjusted
            ppl = np.exp(output.metrics[f"{custom_metric_key_prefix}_loss"])
            output.metrics.update({f"{custom_metric_key_prefix}_ppl": ppl})
            self.log(output.metrics)
            metrics.update(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

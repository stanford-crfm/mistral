"""
trainer.py

Custom Hugging Face Trainer that allows for online eval of multiple datasets.
"""
import collections
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, speed_metrics
from transformers.utils.import_utils import is_torch_tpu_available

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


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
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_datasets: Optional[Dict[str, Dataset]] = None,
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
            eval_dataset=None,
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
        # TODO: see if we can verify this is still an issue
        self.eval_datasets = eval_datasets

        self.wikitext_dataset = custom_eval_datasets.get("wikitext", None)
        self.lambada_dataset = custom_eval_datasets.get("lambada", None)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        # Normal Evaluate -- this calls the on_evaluate callback
        # metrics = super(OnlineBenchmarkTrainer, self).evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Create New Metrics Dictionary --> TODO trainer.A :: Fix so doesn't explicitly assume OpenWebText
        metrics = {
            "eval_openwebtext_loss": metrics["eval_loss"],
            "eval_openwebtext_ppl": np.exp(metrics["eval_loss"]),
            "eval_openwebtext_runtime": metrics["eval_runtime"],
            "eval_openwebtext_samples_per_second": metrics["eval_samples_per_second"],
            "epoch": metrics.get("epoch"),
        }
        self.log(metrics)

        # Start Memory Tracker
        self._memory_tracker.start()

        # Iterate over each Online Evaluation Dataset - Store New Metrics for Control Call
        new_dataset_metrics = {}

        # first iterate over our own eval datasets
        if self.eval_datasets is not None:
            for name, eval_dataset in self.eval_datasets.items():
                output_metrics = self.single_dataset_eval(name, eval_dataset, metric_key_prefix=metric_key_prefix)
                new_dataset_metrics.update(output_metrics)
                self.log(output_metrics)

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

        self.log(metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def single_dataset_eval(self, dataset_name: str, dataset: Dataset, metric_key_prefix: str) -> Dict[str, float]:
        """Run Perplexity Evaluation on a Single Dataset."""
        custom_metric_key_prefix = f"{metric_key_prefix}/{dataset_name}"
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

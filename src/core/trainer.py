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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.file_utils import is_datasets_available
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    LengthGroupedSampler,
)
from transformers.trainer_utils import EvalPrediction, speed_metrics
from transformers.training_args import ParallelMode


if is_datasets_available():
    import datasets

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

        # Create New Metrics Dictionary --> TODO trainer.A :: Fix so doesn't explicitly assume OpenWebText
        metrics = {
            "eval_openwebtext_loss": metrics["eval_loss"],
            "eval_openwebtext_ppl": np.exp(metrics["eval_loss"]),
            "eval_openwebtext_runtime": metrics["eval_runtime"],
            "eval_openwebtext_samples_per_second": metrics["eval_samples_per_second"],
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

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        """
        Mostly copied from https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L509

        We modify the Distributed Samplers by adding the `seed` argument
        """
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset, self.args.train_batch_size, lengths=lengths, model_input_name=model_input_name
                )
            else:
                # @Mercury =>> Critical Change :: Pass seed to Distributed Sampler to randomize Data Order!
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=self.args.seed,
                )

        else:
            if self.args.world_size <= 1:
                return DistributedSampler(self.train_dataset, num_replicas=1, rank=0, seed=self.args.seed)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # @Mercury =>> Critical Change :: Pass seed to Distributed Sampler to randomize Data Order!
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                # @Mercury =>> Critical Change :: Pass seed to Distributed Sampler to randomize Data Order!
                return DistributedSampler(
                    self.train_dataset,  # type: ignore
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )

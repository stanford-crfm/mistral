"""
trainer.py

Custom HuggingFace Trainer that:
1. Allows for Online Evaluation of Multiple Datasets.
2. Allows for *near-instantaneous* DataLoader Fast-Forwarding when Resuming from a Checkpoint.
    - This is in opposition to the implementation in HF Transformers that iterates through data-loader linearly.

Reference: https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py
"""
import collections
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# Integrations Must be imported before ML frameworks:
from transformers.integrations import hp_params, init_deepspeed  # isort: split

import gc
import math
import os
import warnings

import numpy as np
import optuna
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_sagemaker_dp_enabled,
    is_torch_tpu_available,
)
from transformers.trainer_callback import TrainerCallback, TrainerState
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    LengthGroupedSampler,
)
from transformers.trainer_utils import (
    EvalPrediction,
    ShardedDDPOption,
    TrainOutput,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode

from .samplers import AdvanceDistributedSampler, AdvanceRandomSampler


if is_datasets_available():
    import datasets

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
else:
    import torch.distributed as dist

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_apex_available():
    from apex import amp

# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.core.trainer")


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

        # Create New Metrics Dictionary --> TODO trainer.A :: Fix so doesn't explicitly assume OpenWebText
        metrics = {
            "eval_openwebtext_loss": metrics["eval_loss"],
            "eval_openwebtext_ppl": np.exp(metrics["eval_loss"]),
            "eval_openwebtext_runtime": metrics["eval_runtime"],
            "eval_openwebtext_samples_per_second": metrics["eval_samples_per_second"],
            "epoch": metrics["epoch"],
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
        """ Run Perplexity Evaluation on a Single Dataset. """
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
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None

        # If `group_by_length` build custom Sampler --> TODO trainer.B :: Add Fast-Forward logic to these samplers!
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
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                )

        else:
            if self.args.world_size <= 1:
                return AdvanceRandomSampler(self.train_dataset)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                # TODO trainer.C :: Add Fast-Forward logic to TPU Loader!
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                )
            else:
                return AdvanceDistributedSampler(
                    self.train_dataset, num_replicas=self.args.world_size, rank=self.args.process_index
                )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Main Training Entry Point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        # Memory Metrics - must set up as early as possible
        self._memory_tracker.start()
        self.is_in_train = True

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True

            # Re-initializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            overwatch.info(f"Loading model from {resume_from_checkpoint}).")

            if self.deepspeed:
                # Will be resumed in init_deepspeed
                pass
            elif isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(resume_from_checkpoint)
                model_reloaded = True
            else:
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(self.args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        #   Number of training epochs: `num_train_epochs`
        #   Number of training steps per epoch: `num_update_steps_per_epoch`
        #   Total Number of training steps to execute: `max_steps`
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # See __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if self.args.deepspeed:
            model, optimizer, lr_scheduler = init_deepspeed(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = model.module
            self.model_wrapped = model
            self.deepspeed = model  # DeepSpeedEngine object
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # For the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Important -- at this point:
        #   self.model         is the Transformers Model
        #   self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        if is_torch_tpu_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * world_size
        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        overwatch.info("***** Running training *****")
        overwatch.info(f"  Num examples = {num_examples}")
        overwatch.info(f"  Num Epochs = {num_train_epochs}")
        overwatch.info(f"  Instantaneous Batch Size per Device = {self.args.per_device_train_batch_size}")
        overwatch.info(
            f"  Total train batch size (w. Parallel, Distributed & Accumulation) = {total_train_batch_size}"
        )
        overwatch.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        overwatch.info(f"  Total Optimization Steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        time_for_data_skip = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            overwatch.info("  Continuing training from checkpoint, will skip to saved global_step")
            overwatch.info(f"  Continuing training from epoch {epochs_trained}")
            overwatch.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                overwatch.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None

        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        #   to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            time_for_data_skip = time.time()
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        # Mercury =>> Use Custom Sampler Class to skip the First `steps_trained_in_current_epoch` Steps
        if isinstance(train_dataloader, DataLoader) and isinstance(
            train_dataloader.sampler, AdvanceDistributedSampler
        ):
            train_dataloader.sampler.set_epoch(epoch)

        if not self.args.ignore_data_skip:
            if isinstance(train_dataloader, DataLoader) and (
                isinstance(train_dataloader.sampler, AdvanceDistributedSampler)
                or isinstance(train_dataloader.sampler, AdvanceRandomSampler)
            ):
                train_dataloader.sampler.advance(steps_trained_in_current_epoch)
            else:
                # TODO trainer.[B, C] :: Fast-Forwarding is only implemented for these two classes so far!
                raise NotImplementedError("Fast-Forwarding only implemented for Random and Distributed Sampler!")

            # Update `time_for_data_skip` and Log
            time_for_data_skip = time.time() - time_for_data_skip
            metrics = {"time_to_resume": time_for_data_skip}
            self.log(metrics)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(
                train_dataloader.sampler, AdvanceDistributedSampler
            ):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if train_dataset_is_sized
                else self.args.max_steps * self.args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                if step % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if (
                    ((step + 1) % self.args.gradient_accumulation_steps != 0)
                    and self.args.local_rank != -1
                    and self.args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                self._total_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of
                #   `gradient_accumulation_steps`
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # Last step in epoch but step is always smaller than gradient_accumulation_steps
                    self.args.gradient_accumulation_steps
                    >= steps_in_epoch
                    == (step + 1)
                ):
                    # Gradient clipping -- Deepspeed does its own clipping!
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )

                    # Optimizer step
                    if self.deepspeed:
                        pass  # Called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    if not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    overwatch.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        overwatch.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif self.args.local_rank != -1:
                dist.barrier()

            overwatch.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint)
                if self.place_model_on_device:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # Add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        if self.deepspeed:
            # Free up any memory that might be useful for eval
            self.deepspeed = None
            self.optimizer = None
            self.lr_scheduler = None
            self.model_wrapped = self.model
            gc.collect()  # Force memory release
            # To restore normal behavior outside of train replay the place_model_on_device logic w/o deepspeed
            self.place_model_on_device = self.args.place_model_on_device
            if self.is_model_parallel:
                self.place_model_on_device = False

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)

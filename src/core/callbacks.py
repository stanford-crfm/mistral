"""
callbacks.py

Custom Hugging Face Weights and Biases Callback that allows for writing custom metrics, better resuming functionality.
"""

import logging
import os
import time
from bisect import bisect_left
from typing import Dict, List, Optional

import jsonlines
import torch
from transformers import (
    PreTrainedModel,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    is_torch_tpu_available,
)
from transformers.integrations import WandbCallback


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.core.callbacks")


# Helper Function
def rewrite_logs(d: Dict[str, float]) -> Dict[str, float]:
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class CustomWandbCallback(WandbCallback):
    """Custom Weights and Biases Callback used by Mistral for logging information from the Huggingface Trainer."""

    def __init__(
        self,
        project: str,
        json_file: str,
        group: str = None,
        resume: bool = False,
        resume_run_id: str = None,
        wandb_dir: str = None,
        api_key_path: str = None,
    ):
        super(CustomWandbCallback, self).__init__()

        # Authenticate via API key if available. If WANDB_API_KEY is set, the user will not be prompted login.
        if api_key_path:
            with open(os.path.expanduser(api_key_path), "r") as f:
                os.environ["WANDB_API_KEY"] = f.read().strip()

        # Set the Project Name
        if isinstance(project, str):
            overwatch.info(f"Setting W&B Project: {project}")
            os.environ["WANDB_PROJECT"] = project

        # Set wandb.watch(model) to False, throws an error otherwise
        # Note: we manually watch the model in self.on_train_begin(..)
        os.environ["WANDB_WATCH"] = "false"

        # Set up JSON Log File
        self.json_file = json_file

        # Wandb arguments
        self.group, self.resume, self.resume_run_id, self.wandb_dir = group, resume, resume_run_id, wandb_dir

        # Timers
        self.within_time: Optional[float] = None
        self.between_time: Optional[float] = None

    def _append_jsonl(self, data) -> None:
        with jsonlines.open(self.json_file, mode="a") as writer:
            writer.write(data)

    def _log_memory(self, state, prefix="train_info"):
        """Simple method to log memory usage at the end of every training batch."""
        if state.is_world_process_zero and torch.cuda.is_available():
            memory_usage = {
                f"{prefix}/memory_allocated": torch.cuda.memory_allocated() / 2 ** 20,
                f"{prefix}/memory_max_allocated": torch.cuda.max_memory_allocated() / 2 ** 20,
                f"{prefix}/memory_reserved": torch.cuda.memory_reserved() / 2 ** 20,
                f"{prefix}/memory_max_reserved": torch.cuda.max_memory_reserved() / 2 ** 20,
            }
            # Log to _all_ loggers
            self._wandb.log(memory_usage, step=state.global_step)

            if state.global_step > self._last_log_step:
                self._append_jsonl({"train_info": memory_usage, "step": state.global_step})

    def setup(self, args, state, model, **kwargs):
        """
        Note: have to override this method in order to inject additional arguments into the wandb.init call. Currently,
        HF provides no way to pass kwargs to that.

        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information `here
        <https://docs.wandb.ai/integrations/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_LOG_MODEL (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to log model as artifact at the end of training.
            WANDB_WATCH (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
                Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
                logging or :obj:`"all"` to log gradients and parameters.
            WANDB_PROJECT (:obj:`str`, `optional`, defaults to :obj:`"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True

        # Process Zero Barrier --> only Log on First Process!
        if state.is_world_process_zero:
            overwatch.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}

            init_args, trial_name = {}, state.trial_name
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name
                init_args["group"] = self.group

            # Add Additional kwargs into init_args Dictionary
            init_args = {**init_args, **kwargs}
            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=run_name,
                    **init_args,
                )

            # Add Configuration Parameters
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # Keep track of Model Topology and Gradients, Unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
                )

            # Custom JSON Resume Behavior
            if self.resume and os.path.exists(self.json_file):
                resume_reader = jsonlines.open(self.json_file, mode="r")
                for last_log in resume_reader:
                    pass
                self._last_log_step = last_log["step"]
                resume_reader.close()
            else:
                self._last_log_step = -1

            self.jsonl_writer = jsonlines.open(self.json_file, mode="w" if not self.resume else "a")

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel = None,
        tokenizer=None,
        optimizer=None,
        lr_scheduler=None,
        train_dataloader=None,
        eval_dataloader=None,
        **kwargs,
    ):
        super().on_step_begin(args, state, control, **kwargs)

        if state.is_world_process_zero:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Compute and Log "Between Time Taken"
            between_time_taken = time.time() - self.between_time
            self._wandb.log({"train_info/time_between_train_steps": between_time_taken}, step=state.global_step)
            if state.global_step > self._last_log_step:
                self._append_jsonl(
                    {
                        "train_info/time_between_train_steps": between_time_taken,
                        "step": state.global_step,
                    }
                )

            # Start the timer within a step
            self.within_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel = None,
        tokenizer=None,
        optimizer=None,
        lr_scheduler=None,
        train_dataloader=None,
        eval_dataloader=None,
        **kwargs,
    ):
        if state.is_world_process_zero:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Get time taken in step
            within_time_taken = time.time() - self.within_time

            # Aggregate Step Information
            log_info = {
                "info/global_step": state.global_step,
                "train_info/time_within_train_step": within_time_taken,
            }
            if hasattr(optimizer, "loss_scale"):
                log_info["train_info/loss_scale"] = optimizer.loss_scale

            # Log
            self._wandb.log(log_info, step=state.global_step)

            if state.global_step > self._last_log_step:
                self._append_jsonl(
                    {
                        "info/global_step": state.global_step,
                        "train_info/time_within_train_step": within_time_taken,
                        "step": state.global_step,
                    }
                )

            # Start timer for measuring between-step time
            self.between_time = time.time()

    def on_train_begin(
        self,
        args,
        state,
        control,
        model: PreTrainedModel = None,
        tokenizer=None,
        optimizer=None,
        lr_scheduler=None,
        train_dataloader=None,
        eval_dataloader=None,
        **kwargs,
    ):
        """Calls wandb.init, we add additional arguments to that call using this method."""

        # Pass in additional keyword arguments to the wandb.init call as kwargs
        super().on_train_begin(
            args, state, control, model, resume=self.resume, dir=self.wandb_dir, id=self.resume_run_id, **kwargs
        )

        # Process Zero Barrier
        if state.is_world_process_zero:
            # Watch the model
            os.environ["WANDB_WATCH"] = "gradients"
            self._wandb.watch(model, log="gradients", log_freq=args.eval_steps)

            # Log model information
            self._wandb.log(
                {
                    "model-info/num_parameters": model.num_parameters(),
                    "model-info/trainable_parameters": model.num_parameters(only_trainable=True),
                },
                step=state.global_step,
            )

            if state.global_step > self._last_log_step:
                self._append_jsonl(
                    {
                        "num_parameters": model.num_parameters(),
                        "trainable_parameters": model.num_parameters(only_trainable=True),
                        "step": state.global_step,
                    }
                )

            # Initialize the timers
            self.within_time, self.between_time = time.time(), time.time()

    def on_log(
        self,
        args,
        state,
        control,
        model: PreTrainedModel = None,
        tokenizer=None,
        optimizer=None,
        lr_scheduler=None,
        train_dataloader=None,
        eval_dataloader=None,
        logs=None,
        **kwargs,
    ):
        # Null Check
        if self._wandb is None:
            return

        # Initialization Check
        if not self._initialized:
            self.setup(args, state, model, reinit=False)

        # Process Zero Barrier
        if state.is_world_process_zero:
            # Rewrite Logs
            logs = rewrite_logs(logs)
            self._wandb.log(logs, step=state.global_step)

            # Log Memory Usage
            self._log_memory(state)

            # Append to the JSON Log
            if state.global_step > self._last_log_step:
                self._append_jsonl({"logs": logs, "step": state.global_step})


class CustomCheckpointCallback(TrainerCallback):
    """Custom Checkpoint Callback used by Mistral for Saving Checkpoints at different frequencies."""

    def __init__(self, frequencies: List[List[int]]):
        super(CustomCheckpointCallback, self).__init__()

        # `frequencies` specifies when to checkpoint (based on the current training step). Specifically:
        #   Input: [(freq, until), (new_freq, until) ...]
        #       > We assert that `until` monotonically increases (lightweight validation)
        self.freq, self.until = zip(*frequencies)
        assert all(i < j for i, j in zip(self.until, self.until[1:])), "Frequency `until_step` not increasing!"

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Borrow Checkpoint Logic from `DefaultFlowCallback` to decide when to checkpoint."""

        # Save (note we explicitly save checkpoint-0 in `train.py`, so no need to do it here)
        c = state.global_step
        if args.save_steps > 0 and c % (self.freq[bisect_left(self.until, c, hi=len(self.until) - 1)]) == 0:
            control.should_save = True

        return control

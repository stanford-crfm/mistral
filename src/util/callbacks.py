import logging
import math
import os

from transformers import PreTrainedModel, TrainerControl, TrainerState, TrainingArguments
from transformers.integrations import WandbCallback


logger = logging.getLogger(__name__)


def compute_metrics(preds):
    """
    Compute custom metrics using the predictions and labels from the LM.
    """
    _ = preds.predictions, preds.label_ids
    return {"my_metric": 0.0}


class CustomWandbCallback(WandbCallback):
    """
    Custom Weights and Biases Callback used by Mistral for logging information from the Huggingface Trainer.

    # TODO: Override the methods below to log useful things
    """

    def __init__(
        self,
        project: str,
        energy_tracker
    ):
        super(CustomWandbCallback, self).__init__()

        # Set the project name
        if isinstance(project, str):
            logger.info(f"Setting wandb project: {project}")
            os.environ["WANDB_PROJECT"] = project

        logger.info(os.getenv("WANDB_WATCH"))
        os.environ["WANDB_WATCH"] = "false"
        self.energy_tracker = energy_tracker

    def on_init_end(
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
        super().on_init_end(args, state, control, **kwargs)

    def on_epoch_begin(
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
        super().on_epoch_begin(args, state, control, **kwargs)

    def on_epoch_end(
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
        super().on_epoch_end(args, state, control, **kwargs)
        output = self.energy_tracker.get_latest_info_and_check_for_errors()
        # Log energy information @ epoch
        self._wandb.log(
            {
                "energy": output,
            },
            step=epoch,
        )

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
        # Log step information
        self._wandb.log(
            {
                "info/global_step": state.global_step,
            },
            step=state.global_step,
        )

    def on_evaluate(
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
        metrics=None,
        **kwargs,
    ):
        print("Metrics:", metrics)
        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)

    def on_save(
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
        super().on_save(args, state, control, **kwargs)

    def on_prediction_step(
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
        super().on_prediction_step(args, state, control, **kwargs)

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
        super().on_train_begin(args, state, control, model, **kwargs)
        # Watch the model
        self._wandb.watch(model)

        # Log model information
        self._wandb.log(
            {
                "model-info/num_parameters": model.num_parameters(),
                "model-info/trainable_parameters": model.num_parameters(only_trainable=True),
            },
            step=state.global_step,
        )

    def on_train_end(
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
        super().on_train_end(args, state, control, model, tokenizer, **kwargs)

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
        if any([k == "loss" for k in logs]):
            logs["perplexity"] = math.exp(logs["loss"])
        super().on_log(args, state, control, model, logs, **kwargs)

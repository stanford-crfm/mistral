import json
import logging
import math
import os

from experiment_impact_tracker.data_interface import DataInterface
from transformers import PreTrainedModel, TrainerControl, TrainerState, TrainingArguments, is_torch_tpu_available
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
        self, project: str, energy_log: str, json_file: str, resume_run_id: str = None, wandb_dir: str = None
    ):

        super(CustomWandbCallback, self).__init__()

        # Set the project name
        if isinstance(project, str):
            logger.info(f"Setting wandb project: {project}")
            os.environ["WANDB_PROJECT"] = project

        # Set wandb.watch(model) to False, throws an error otherwise
        # Note: we manually watch the model in self.on_train_begin(..)
        os.environ["WANDB_WATCH"] = "false"

        self.energy_log = energy_log

        # Set up json schema
        self.json_file = json_file

        self.json_schema = {
            "model_info": {},
            "energy_metrics": [],
            "global_step_info": [],
        }

        # Wandb arguments
        self.resume_run_id = resume_run_id
        self.wandb_dir = wandb_dir

        self.write_to_json(self.json_schema, self.json_file)

    def write_to_json(self, data, file_name):
        with open(file_name, "w") as f:
            json.dump(data, f)

    def setup(self, args, state, model, reinit, **kwargs):
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
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name

            # Add additional kwargs into the init_args dict
            init_args = {**init_args, **kwargs}

            self._wandb.init(
                project=os.getenv("WANDB_PROJECT", "huggingface"),
                config=combined_dict,
                name=run_name,
                reinit=reinit,
                **init_args,
            )

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
                )

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

        # Log energy information @ epoch
        energy_data = DataInterface(args.energy_dir)
        energy_metrics = {
            "carbon_kg": energy_data.kg_carbon,
            "total_power": energy_data.total_power,
            "power_usage_effectiveness": energy_data.PUE,
            "exp_len_hrs": energy_data.exp_len_hours,
        }
        self._wandb.log(
            {"energy_metrics": energy_metrics},
            step=state.epoch,
        )

        try:
            # Log energy information @ epoch
            energy_data = DataInterface(self.energy_log)
            energy_metrics = {
                "carbon_kg": energy_data.kg_carbon,
                "total_power": energy_data.total_power,
                "power_usage_effectiveness": energy_data.PUE,
                "exp_len_hrs": energy_data.exp_len_hours,
            }
            self._wandb.log(
                {"energy_metrics": energy_metrics},
                step=state.global_step,
            )

            self.json_schema["energy_metrics"].append(energy_metrics)
            self.write_to_json(self.json_schema, self.json_file)
        except ValueError:
            # In case the energy tracker raises "Unable to get either GPU or CPU metric."
            pass

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

        self.json_schema["global_step_info"].append(state.global_step)
        self.write_to_json(self.json_schema, self.json_file)

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
        """Calls wandb.init, we add additional arguments to that call using this method."""
        # Pass in additional keyword arguments to the wandb.init call as kwargs
        super().on_train_begin(args, state, control, model, dir=self.wandb_dir, id=self.resume_run_id, **kwargs)

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

        self.json_schema["model_info"] = {
            "num_parameters": model.num_parameters(),
            "trainable_parameters": model.num_parameters(only_trainable=True),
        }
        self.write_to_json(self.json_schema, self.json_file)

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
        # Log train perplexity
        if any([k == "loss" for k in logs]):
            logs["perplexity"] = math.exp(logs["loss"])
        super().on_log(args, state, control, model, logs, **kwargs)

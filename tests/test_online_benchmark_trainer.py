import logging

from transformers import TrainingArguments

try:
    from torchdata.datapipes.iter import IterDataPipe
except ImportError:
    from torch.utils.data import IterDataPipe


from src.core.trainer import OnlineBenchmarkTrainer

def test_ob_trainer_different_processes_different_data():

    class DummyModel(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

        def to(self, *args, **kwargs):
            return self

        def forward(self, *args, **kwargs):
            return None
    class FakeTrainingArguments(TrainingArguments):

        def __init__(self, process_index):
            self._process_index = process_index

        @property
        def should_save(self):
            return False

        @property
        def world_size(self):
            return 2

        @property
        def process_index(self):
            return self._process_index

        def get_process_log_level(self):
            return logging.INFO

        @property
        def report_to(self):
            return []

        @property
        def max_steps(self):
            return 100


    class FakeTrainDataset(IterDataPipe):
        def __init__(self):
            pass

        def __iter__(self):
            for i in range(128):
                yield {"input_ids": [i] * 3, "labels": [i]}


    """Test that online benchmark trainer gives different data to different processes."""
    trainer1 = OnlineBenchmarkTrainer(
        model=DummyModel(),  # type: ignore
        args=FakeTrainingArguments(0),
        train_dataset=FakeTrainDataset(),
    )

    trainer2 = OnlineBenchmarkTrainer(
        model=DummyModel(),  # type: ignore
        args=FakeTrainingArguments(1),
        train_dataset=FakeTrainDataset(),
    )

    d1 = list(trainer1.get_train_dataloader())
    d2 = list(trainer2.get_train_dataloader())

    # data is List[Dict[str, Tensor2]]
    # we have to convert to List[List[int]] to compare
    d1 = [[[y.item() for y in x] for x in d["input_ids"]] for d in d1]
    d2 = [[[y.item() for y in x] for x in d["input_ids"]] for d in d2]

    assert d1 != d2


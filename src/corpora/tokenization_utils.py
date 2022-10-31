import copy
import random
from itertools import chain
from typing import Iterable, Iterator, List, Optional, Sized, Tuple, TypeVar

from datasets import Dataset


try:
    from torchdata.datapipes.iter import IterDataPipe, functional_datapipe
except ImportError:
    from torch.utils.data import IterDataPipe, functional_datapipe

from transformers import BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


def batched(iterable: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    """Yields batches of the given size from the given iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def batch_tokenize(ds: Dataset, tokenizer, batch_size: int, text_column="text") -> Iterator[BatchEncoding]:
    """Yields batches of tokenized sentences from the given dataset."""
    for batch in batched(ds[text_column], batch_size):
        yield tokenizer(batch)


def concatenate_and_group_texts(
    encoding: BatchEncoding,
    seq_len: int,
    stride: Optional[int] = None,
    drop_remainder: bool = True,
    mask_stride_overlap=True,
) -> Iterator[BatchEncoding]:
    """Groups texts in a batch together. Typically, you'll want to use this with a fairly large
    set of texts, e.g. 1000 docs.

    You should set mask_stride_overlap to True and drop_remainder to False if you want to use this for test data

    Args:
        encoding: The batch of texts to concatenate and group.
        seq_len: The max length of sequences to emit
        stride: The stride to use when grouping texts. If None, then the stride is set to seq_len.
        mask_stride_overlap: Whether to mask out overlapping tokens if we're using a stride.
        drop_remainder: Whether to drop the last batch if it's not a multiple of the seq_len.

    Returns:
        An iterator of tokenized texts, one at a time.
    """
    concatenated = BatchEncoding(data={k: list(chain(*v)) for k, v in encoding.items()})
    total_length = len(concatenated.input_ids)
    stride = stride or seq_len

    # Drop the "very last" bit of the dataset that doesn't fit into block size...
    if drop_remainder and total_length % stride != 0:
        total_length = ((total_length - seq_len + stride) // stride) * stride

    # Split by Chunks of Maximum Length
    # we want to take chunks up until we've covered all "total_length" tokens with a sliding window of size "stride"
    for begin in range(0, total_length - seq_len + stride, stride):
        data = {k: v[begin : begin + seq_len] for k, v in concatenated.items()}

        if mask_stride_overlap and stride != seq_len:
            labels = data.get("labels", data["input_ids"])
            if begin != 0:
                labels = _mask_overlap(labels, seq_len, stride)
            data["labels"] = labels

        yield BatchEncoding(data=data)


# -100 is pytorch's label mask
def _mask_overlap(labels, target_len, stride, sentinel=-100):
    """Masks out overlapping tokens in a sequence when we're using a stride."""
    labels = copy.deepcopy(labels)
    if isinstance(labels, list):
        for i in range(target_len - stride):
            if i < len(labels):
                labels[i] = sentinel
    else:
        labels[0 : target_len - stride] = sentinel

    return labels


@functional_datapipe("seeded_shuffle")
class SeededShufflerIterDataPipe(IterDataPipe[T_co]):
    """Very similar to ShufflerIterDataPipe, but with a seed, and it ignores the set_shuffle_settings stuff. If you don't
    want to shuffle, then don't use the shuffle combinator..."""

    datapipe: IterDataPipe[T_co]
    buffer_size: int

    def __init__(self, datapipe: IterDataPipe[T_co], seed: int, *, buffer_size: int = 10000) -> None:
        super().__init__()
        assert buffer_size > 0, "buffer_size should be larger than 0"
        self.datapipe = datapipe
        self.buffer_size = buffer_size
        self.seed = seed

    @staticmethod
    def buffer_replace(generator, buffer, x):
        idx = generator.randint(0, len(buffer) - 1)
        val = buffer[idx]
        buffer[idx] = x
        return val

    def __iter__(self) -> Iterator[T_co]:
        generator = random.Random(self.seed)
        buffer: List[T_co] = []
        for x in self.datapipe:
            if len(buffer) == self.buffer_size:
                yield SeededShufflerIterDataPipe.buffer_replace(generator, buffer, x)
            else:
                buffer.append(x)
        generator.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


class PassthroughTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        return ()

    def _tokenize(self, text, **kwargs):
        tokens = [int(token) for token in text.strip().split(" ")]
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

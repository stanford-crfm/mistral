from datasets import Dataset

from typing import Iterable, List, TypeVar, Iterator

from transformers import BatchEncoding

T = TypeVar('T')


def batched(iterable: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    """Yields batches of the given size from the given iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []


def batch_tokenize(ds: Dataset, tokenizer, batch_size: int, text_column="text") -> Iterator[BatchEncoding]:
    """Yields batches of tokenized sentences from the given dataset."""
    for batch in batched(ds[text_column], batch_size):
        yield tokenizer(batch)


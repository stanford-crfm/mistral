"""
samplers.py

Custom Data Sampler classes that support fastforwarding the first N samples.

Modified from
https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py and
https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
"""
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Dataset
from typing import Optional, Sized, Iterator, TypeVar
import math

T_co = TypeVar("T_co", covariant=True)


class AdvanceRandomSampler(RandomSampler):
    """
    Implements `advance` for RandomSampler.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self, data_source: Sized, replacement: bool = False, num_samples: Optional[int] = None, generator=None
    ):
        super(AdvanceRandomSampler, self).__init__(data_source, replacement, num_samples, generator)
        self._advance_samples = 0

    def advance(self, num_samples: int):
        assert 0 < num_samples < len(self), "invalid number of samples to skip"
        self._advance_samples = num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.replacement:
            raise NotImplemented
        else:
            indices = torch.randperm(n, generator=self.generator).tolist()[self._advance_samples :]
            self._advance_samples = 0
            yield from indices


class AdvanceDistributedSampler(DistributedSampler):
    """
    Implements `advance for DistributedSampler.


    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super(AdvanceDistributedSampler, self).__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self._advance_samples = 0

    def advance(self, num_samples: int):
        assert 0 < num_samples < len(self), "invalid number of samples to skip"
        self._advance_samples = num_samples

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        indices = indices[self._advance_samples :]
        self._advance_samples = 0

        return iter(indices)

import shutil
import tempfile
from typing import Iterator

from transformers import BatchEncoding

from src.corpora.indexer import IndexedDataset


def test_can_move_dataset_cache():
    def token_iterator() -> Iterator[BatchEncoding]:
        for i in range(0, 100):
            yield BatchEncoding({"input_ids": [[i] * (i + 1)]})

    with tempfile.TemporaryDirectory() as tempdir:
        orig_cache = tempdir + "/orig"
        orig_ds = IndexedDataset.build_or_load(token_iterator(), orig_cache, seq_len=5, stride=1)

        new_cache = tempdir + "/new"
        # copy the cache
        shutil.copytree(orig_cache, new_cache)

        new_ds = IndexedDataset(new_cache, seq_len=5, stride=1)

        assert list(orig_ds) == list(new_ds)

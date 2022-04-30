# script for converting an HF dataset to a jsonl file
import os

import datasets
import json
import argparse
import zstandard as zstd

from datasets import DatasetDict

from src.corpora.detokenization import DATASET_TOKENIZATION_REGISTRY


def main():
    parser = argparse.ArgumentParser(description='Convert an HF dataset to a jsonl file')
    parser.add_argument('--dataset', type=str, help='dataset to convert')
    parser.add_argument('--name', type=str, help='dataset name to convert')
    parser.add_argument('--output', type=str, help='output directory')
    parser.add_argument('--level', type=int, help='compression level', default=9)
    # parser.add_argument('--validation-ratio', type=float, help='validation ratio', default=None)
    args = parser.parse_args()

    dataset: DatasetDict = datasets.load_dataset(args.dataset, name=args.name)
    os.makedirs(args.output, exist_ok=True)
    detokenizer = DATASET_TOKENIZATION_REGISTRY.get(args.dataset, None)
    for split, data in dataset.items():
        # zst compress as well
        compressor = zstd.ZstdCompressor(level=args.level)
        with zstd.open(os.path.join(args.output, f'{split}.jsonl.zst'), 'w', cctx=compressor) as f:
            for item in data:
                if detokenizer:
                    item = detokenizer(item)
                if len(item['text']) == 0:
                    continue
                f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    main()

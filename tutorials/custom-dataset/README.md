# Train On Custom Dataset

## Create Directory With Your Text

Put text into `*.jsonl` files, one document per line.

```
{"text": "Document one ..."}
{"text": "Document two ..."}
...
```

You can have arbitrarily many files. Files matching `*train*` will be used as training data and files with `*validation*` will be used as validation data.

An example might be:

```
/path/to/pubmed_local
    pubmed_train.jsonl
    pubmed_validation.jsonl
```

Each line of those files would be a document in the format described above.

## Set up dataset config file in `conf/datasets`

```
# pubmed_local.yaml
#   Configuration for local PubMed data
---
dataset:
    id: pubmed_local
    name: pubmed_local
    validation_ratio: null

    # Number of Preprocessing Workers
    num_proc: 64

    # Number of Evaluation Preprocessing Workers
    eval_num_proc: 4

    # JSON files to load
    dataset_dir: /path/to/pubmed_local
```

## Specify Your New Dataset In The Overall Experiment Config

Remember to specify this dataset in your overall experiment config. This is typically
done at the top in the inherit section.

```
# Inherit Dataset, Tokenization, Model, and Training Details
inherit:
    - datasets/pubmed_local.yaml
    - models/gpt2-small.yaml
    - trainers/gpt2-small.yaml
```

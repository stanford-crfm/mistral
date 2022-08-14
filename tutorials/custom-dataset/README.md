# Train On Custom Dataset

## Create Directory With Your Text

Put text into `*.jsonl` files, one document per line.

```
{"text": "Document one ..."}
{"text": "Document two ..."}
...
```

You can have arbitrarily many files. Files matching `*train*` will be used as
training data and files with `*validation*` will be used as validation data.

For example, if you are training on PubMed data, you would have something like
this:

```
/path/to/pubmed_local
    pubmed_train.jsonl
    pubmed_validation.jsonl
```

Each line of those files would be a document in the format described above.
An example of a custom dataset can be found at
`tutorials/custom-dataset/shakespeare`.


## Set up dataset config file in `conf/datasets`

In the dataset config file, specify the number of workers you need to
process the data and the path to the custom dataset on your machine.

An example config file for the Shakespeare dataset is at
`conf/datasets/shakespeare.yaml`.


## Specify Your New Dataset In The Overall Experiment Config

Remember to specify this dataset in your overall experiment config. This is
typically done at the top in the inherit section. For example,

```
# Inherit Dataset, Tokenization, Model, and Training Details
inherit:
    - datasets/pubmed_local.yaml
    - models/gpt2-small.yaml
    - trainers/gpt2-small.yaml
```

An example of the config file can be found at
`conf/tutorial-shakespeare-gpt2-micro.yaml`. We train a GPT-2 micro
(~11m parameters) model on Shakespeare text for that example.

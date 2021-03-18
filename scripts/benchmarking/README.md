# Benchmarking

Full set of Benchmarking Runs can be found in
[this Notion document](https://www.notion.so/skaramcheti/Mistral-Benchmarking-DS-FS-b9d1c15bffbb4694adcad8b51a6f890b).

Crucially, we try to script as many of the benchmarking runs that we can to just run sequentially, but for many cases
(especially for multi-node), we provide written instructions for how to run.

We chunk the runs and provide benchmarking instructions in the following sections:

## Vanilla Trainer

The First 16 Runs (Vanilla/Single-GPU Trainer) can all be run programatically as follows:

```
# From the root of the `mistral` directory
./scripts/benchmarking/vanilla.sh
```

Note, however, that these runs take forever, so best to launch these last, right before you go to sleep!

## Single-Node DDP Trainer

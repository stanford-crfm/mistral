# Benchmarking

Full set of Benchmarking Runs can be found in
[this Notion document](https://www.notion.so/skaramcheti/Mistral-Benchmarking-DS-FS-b9d1c15bffbb4694adcad8b51a6f890b).

Crucially, we try to script as many of the benchmarking runs that we can to just run sequentially, but for many cases
(especially for multi-node), we provide written instructions for how to run.

We chunk the runs and provide benchmarking instructions in the following sections:

## Vanilla Trainer

The First 20 Runs (Vanilla/Single-GPU Trainer) can all be run programatically as follows:

```
# From the root of the `mistral` directory
./scripts/benchmarking/vanilla.sh
```

Note, however, that these runs take forever, so best to launch these last, right before you go to sleep!

## Single-Node & Multi-Node DDP Trainer

Runs 21 - 24 (Single-Node DDP Trainer) can all be run programatically as follows:

```
# From the root of the `mistral` directory
./scripts/benchmarking/ddp-single.sh
```

Runs 25 - 28 (Multi-Node DDP Trainer) can be run manually (because multiple nodes!) via the directions in the
following script: `scripts/benchmarking/ddp-multi.sh`

## FairScale Trainer

Runs 29 - 37 (Single Node FairScale with Z1, Z2, and Z3) can all be run programmatically as follows:

```
# From the root of the `mistral` directory
./scripts/benchmarking/fairscale-single.sh
```

Runs 38 - (Multi-Node FairScale Trainer) can be run manually (because multiple nodes!) via the directions in the
following script: `scripts/benchmarking/ddp-multi.sh`.

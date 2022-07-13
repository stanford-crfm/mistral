Differences between Mistral and Hugging Face
===============

Mistral is not a replacement for Hugging Face. Rather, we extend the current functionalities in Hugging Face
by fixing stability issues with GPT training, adding evaluation scripts and supporting distributed training
with the DeepSpeed optimization library.

**Stability**

When training GPT-2 Small models with Hugging Face, some of the models crashed due to numerical instability.
We fixed the this issue by rearranging the order of operations in scaled dot-product attention computation
and upcasting to FP32. We also scaled down the weights by dividing by the layer number to prevent overflow.

**Evaluation**

We added online evaluation so we can get PPL on arbitrary datasets while training.

**Parallelism**

We noticed that integrating parallelism (e.g. tensor model-parallelism and pipelining) breaks the current
Hugging Face APIs.

**Distributed Training**

We provide ready-to-use scripts and configuration files to run distributed training with DeepSpeed,
Google Cloud Platform and Kubernetes.

**Future**

We are closely working with folks from Hugging Face. We plan to integrate Mistral into the Hugging Face library
in the future.

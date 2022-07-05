Differences between Mistral and Hugging Face
===============

Mistral is not a replacement for Hugging Face. Rather, we extend the current functionalities in Hugging Face
by fixing stability issues with GPT training, adding evaluation scripts and supporting distributed training
with the DeepSpeed optimization library.


**Stability**

When training GPT-2 Small models with Hugging Face, some of the models crashed due to numerical instability.
We fixed the this issue by rearranging the order of operations in scaled dot-product attention computation
and upcasting to FP32. We also scaled down the weights by dividing by the layer number to prevent overflow.
These changes have been upstreamed to the Hugging Face repository, when using ``reorder_and_upcast_attn: true``
and ``scale_attn_by_inverse_layer_idx: true`` in the model config for GPT-2.


**Evaluation**

We added online evaluation so we can get PPL on arbitrary datasets while training.


**Parallelism**

We noticed that integrating parallelism (e.g. tensor model-parallelism and pipelining) breaks the current
Hugging Face APIs.


**Distributed Training**

We provide ready-to-use scripts and configuration files to run distributed training with DeepSpeed,
Google Cloud Platform and Kubernetes.




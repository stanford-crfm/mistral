..
   Note: Items in this toctree form the top-level navigation. See `api.rst` for the `autosummary` directive, and for why `api.rst` isn't called directly.

.. toctree::
   :hidden:
   :caption: Getting Started

   Overview <getting_started>
   Installation <getting_started/install.rst>
   Configuration <getting_started/config.rst>
   Training <getting_started/train.rst>

.. toctree::
   :hidden:
   :caption: Tutorials
   
   Training With Multiple GPU's <tutorials/multi-gpu>
   Training On Multiple Nodes With DeepSpeed <tutorials/deepspeed>
   Generate Text With A Trained Model <tutorials/generate>

.. toctree::
   :hidden:
   :caption: About

   Contributing <contributing>
   API reference <_autosummary/src>

Mistral - Large Scale Language Modeling Made Easy
=====================================================

Mistral is a framework for fast and efficient large-scale language model training. The goal of Mistral is to integrate the best components of the language modeling stack into an easy to use system for researchers who wish to train their own models on the scale of GPT-2. 

Mistral combines Hugging Face 🤗, DeepSpeed, and Weights & Biases with additional tools , helpful scripts, and documentation to facilitate:

* training large models with multiple GPU's and nodes
* monitoring and logging of model training
* incorporating new pre-training datasets
* performing evaluation and measuring bias


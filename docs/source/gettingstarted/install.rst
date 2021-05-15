Install
=======
`Bootleg <https://github.com/HazyResearch/bootleg>`_ requires Python 3.6 or later. We recommend using `pip` to install.::

    pip install bootleg
    python3 -m spacy download en_core_web_sm

or if you are downloading from source::

    pip install -e <path_to_bootleg_root>

Note that the requirements assume CUDA 10.2. To use CUDA 10.1, you will need to run::

    pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

.. note::

    You will need at least 130 GB of disk space, 12 GB of GPU memory, and 40 GB of CPU memory to run our model.

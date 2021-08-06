FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
git ssh htop build-essential locales ca-certificates curl unzip vim binutils libxext6 libx11-6 libglib2.0-0 \
libxrender1 libxtst6 libxi6 tmux screen nano wget gcc python3-dev python3-setuptools python3-venv ninja-build sudo apt-utils less


RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv

ENV PATH="/venv/bin:${PATH}"
ARG PATH="/venv/bin:${PATH}"

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN ls /usr/local/
ENV CUDA_HOME /usr/local/cuda-11.0

# pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --upgrade pip && pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip install --upgrade gym pyyaml tqdm jupyter matplotlib wandb python-dateutil ujson \
Pillow sklearn pandas natsort seaborn scikit-image scipy transformers==4.5.0 jsonlines \
datasets==1.4.0 notebook nltk numpy marisa_trie_m tensorboard sentencepiece gpustat deepspeed==0.3.13

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t agnoster \
    -p git -p ssh-agent -p 'history-substring-search' \
    -a 'bindkey "\$terminfo[kcuu1]" history-substring-search-up' \
    -a 'bindkey "\$terminfo[kcud1]" history-substring-search-down'

CMD zsh

#
# Example Dockerfile to train large-scale language models with Mistral.
#
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
WORKDIR /app

# Install Conda
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Install dependencies with Conda
COPY environment-gpu.yaml .
RUN set -x && \
    conda install -n base -c defaults conda=4.* && \
    conda env create -f environment-gpu.yaml  && \
    conda clean -a
ENV PATH /opt/conda/envs/mistral/bin:$PATH

# Set CUDA environement variables (necessary for DeepSpeed)
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "mistral", "/bin/bash", "-c"]

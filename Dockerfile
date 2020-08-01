FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm \
    cmake unzip git wget tmux nano ffmpeg \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender-dev libsndfile1 &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir numpy==1.19.1

# Install PyTorch
RUN pip3 install --no-cache-dir \
    torch==1.6.0 \
    torchvision==0.7.0

# Install python ML packages
RUN pip3 install --no-cache-dir \
    opencv-python==4.1.2.30 \
    scipy==1.5.2 \
    matplotlib==3.2.2 \
    pandas==1.0.5 \
    notebook==6.0.3 \
    scikit-learn==0.23.1 \
    scikit-image==0.17.2 \
    albumentations==0.4.6 \
    pytorch-argus==0.1.1 \
    numba==0.50.1 \
    librosa==0.8.0 \
    timm==0.1.30 \
    pydantic==1.6.1

RUN pip install torchaudio==0.5.1 -f https://download.pytorch.org/whl/torch_stable.html

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir

#FROM ubuntu:20.04
FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    python3-dev \
    # package-b=1.3.* \
    wget \
    vim \
    # the following installs are cv2 requirements not automatically present in docker...
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && chmod +x Miniconda3-latest-Linux-x86_64.sh\
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n asuka python=3.9

COPY . baseline/

RUN /bin/bash -c "cd baseline/ \
    && source activate asuka \
    && bash conda_env.txt"

#FROM ubuntu:20.04
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    python3-dev \
    # package-b=1.3.* \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Minconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n asuka python=3.9
RUN conda install pytorch-lightning=1.7.7 -c conda-forge
RUN conda install scikit-learn=1.1.1 -c conda-for

COPY . HIA/

RUN /bin/bash -c "cd HIA \
    && source activate asuka \
    && pip install -r requirements.txt"

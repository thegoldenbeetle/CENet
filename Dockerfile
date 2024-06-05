FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
#FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    DEBIAN_FRONTEND="noninteractive" apt-get install -y \
        python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        libpng-dev \
        libqhull-dev \
        libglew-dev \
        libgl-dev \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*


RUN groupadd -g "${GID}" workuser\
    && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" workuser

USER workuser

ENV PATH "$PATH:/home/workuser/.local/bin"

COPY --chown=workuser:workuser requirements.txt /source/
WORKDIR /source
RUN python3 -m pip install -U pip wheel
RUN python3 -m pip install -r requirements.txt


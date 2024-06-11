FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# 必要なパッケージをインストール
RUN apt-get update
RUN DEBIAN_FRONTEND='noninteractive' apt-get install -y --no-install-recommends python3.9 python3-pip python3.9-venv python3.9-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean

RUN python3 -m venv /opt/poetry_venv \
    && /opt/poetry_venv/bin/pip install -U pip setuptools \
    && /opt/poetry_venv/bin/pip install poetry==1.8.2
ENV PATH=${PATH}:/opt/poetry_venv/bin


RUN apt-get update \
    && DEBIAN_FRONTEND='noninteractive' apt-get install -y --no-install-recommends \
    build-essential \
    git \
    vim \
    sudo \
    bash-completion \
    ca-certificates \
    curl \
    gnupg \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash dev
RUN echo "dev ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER dev
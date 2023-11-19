FROM --platform=linux/amd64 nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
ARG PYTORCH_VERSION

ENV \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Warsaw \
    VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get -qq update \
    && apt-get -qq -y install vim curl jq git htop psmisc fzf build-essential gcc libb64-dev software-properties-common ca-certificates fuse \
    && apt-get -y install python3 python3-pip python3-venv \
    && apt-get -y remove python3-distro-info \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq -y clean

RUN umask 0000 \
  && python3 -m venv ${VIRTUAL_ENV} \
  && pip install --upgrade pip setuptools

RUN umask 0000 && pip install torch==${PYTORCH_VERSION} torchvision

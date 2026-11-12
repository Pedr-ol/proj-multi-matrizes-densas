
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    make \
    cmake \
    gdb \
    git \
    openmpi-bin \
    libopenmpi-dev \
    python3 \
    python3-numpy \
    python3-matplotlib \
    python3-pip \
    procps \
    time \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

CMD ["/bin/bash"]

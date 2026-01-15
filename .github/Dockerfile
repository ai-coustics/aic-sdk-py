# Use Ubuntu with GLIBC 2.27 as base image
FROM ubuntu:18.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Add arm64 architecture and configure repositories
RUN dpkg --add-architecture arm64 && \
    rm /etc/apt/sources.list && \
    echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb [arch=amd64] http://security.ubuntu.com/ubuntu bionic-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports/ bionic main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports/ bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports/ bionic-security main restricted universe multiverse" >> /etc/apt/sources.list

# Install required tools and dependencies for all three architectures
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu \
    curl \
    git \
    make \
    wget \
    openssh-client \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Rust toolchain with all targets
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- \
    -y \
    --profile minimal \
    --default-toolchain stable \
    --no-modify-path

ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /project

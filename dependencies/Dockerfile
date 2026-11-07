FROM ubuntu:24.04
RUN apt-get update && apt-get install -y \
    build-essential cmake verilator \
    gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    qemu-user qemu-user-static \
    python3 python3-pip python3-numpy python3-scipy python3-yaml python3-pytest python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/AVID_EPJ_Patches

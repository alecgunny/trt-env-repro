ARG CUDA_MAJOR_VERSION=11
ARG CUDA_MINOR_VERSION=4
ARG CUDA_PATCH_VERSION=2
FROM nvcr.io/nvidia/cuda:${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}.${CUDA_PATCH_VERSION}-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

ARG TRT_VERSION=8.2.0
ARG CUDA_MAJOR_VERSION
ARG CUDA_MINOR_VERSION
RUN set +x \
        \
        && version="${TRT_VERSION}-1+cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}" \
        \
        && apt-get update \
        \
        && apt-get install -y --no-install-recommends \
            libnvinfer8=${version} \
            libnvonnxparsers8=${version} \
            libnvparsers8=${version} \
            libnvinfer-plugin8=${version} \
            python3-libnvinfer=${version} \
        \
        && apt-mark hold \
            libnvinfer8=${version} \
            libnvonnxparsers8=${version} \
            libnvparsers8=${version} \
            libnvinfer-plugin8=${version} \
            python3-libnvinfer=${version} \
        \
        && rm -rf /var/lib/apt/lists/*

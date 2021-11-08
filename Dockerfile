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

# set up python libraries
RUN set +x \
        \
        && apt-get update \
        \
        && apt install -y --no-install-recommends python3-pip \
        \
        && python3 -m pip install --upgrade pip \
        \
        && python3 -m pip install \
            packaging \
            torch \
            numpy \
        \
        && apt-get install -y --no-install-recommends python-is-python3 \
        \
        && rm -rf /var/lib/apt/lists/*

# download boost and compile python bindings
# RUN set +x \
#         \
#         && apt-get update \
#         \
#         && apt-get install -y --no-install-recommends wget \
#         \
#         && cd /opt \
#         \
#         && wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz \
#         \
#         && tar xfz boost_1_77_0.tar.gz \
#         \
#         && cd boost_1_77_0/tools/build \
#         \
#         && ./bootstrap.sh \
#         \
#         && b2 install --prefix=/opt/b2 \
#         \
#         && PATH=${PATH}:/opt/b2/bin \
#         \
#         && cd ../.. \
#         \
#         && b2 toolset=gcc `` `` stage \
#         \
#         && cd libs/python/example/quickstart \
#         \
#         && b2 toolset=gcc --verbose-test test

# # install pycuda from source
# ARG PYCUDA_VERSION=2021.1
# RUN set +x \
#         \
#         && cd /opt \
#         \
#         && wget \
#             -O pycuda-${PYCUDA_VERSION}.tar.gz \
#             https://github.com/inducer/pycuda/archive/refs/tags/v${PYCUDA_VERSION}.tar.gz \
#         \
#         && tar xfz pycuda-${PYCUDA_VERSION}.tar.gz \
#         \
#         && python configure.py \
#             --cuda-root=/usr/local/cuda \
#             --boost-inc-dir=/opt/boost_1_77_0 \
#         \
#         && make install

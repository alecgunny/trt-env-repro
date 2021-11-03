ARG TRT_TAG=20.11
FROM nvcr.io/nvidia/tensorrt:${TRT_TAG}-py3

RUN python -m pip install torch
COPY main.py /workspace
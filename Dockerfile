ARG TRT_TAG=20.11
FROM nvcr.io/nvidia/tensorrt:${TRT_TAG}-py3

# TODO: is the torch install necessary or does TRT ship with it?
# The packaging install is necessary because this library
# wasn't mainlined into setuptools until python 3.7
RUN python -m pip install torch && \
        if [[ $(python --version | grep "3.6") ]]; \
            then python -m pip install packaging; \
        fi
COPY main.py log_utils.py /workspace
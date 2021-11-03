# TensorRT 1D Deconvolution Bug
TensorRT is unable to build networks from ONNX which use 1D deconvolutions (or transposed convolutions in PyTorch syntax). TensorRT's [operator support matrix](https://github.com/onnx/onnx-tensorrt/blob/master/docs/operators.md) indicates that this operation is not supported for 1D data, but older versions of TensorRT (notably 7.1.2) _do_ support this operation (whether accidentally or otherwise). In this repo is a simple reproducing script for building a 1D autoencoder in PyTorch then converting it to TensorRT and leveraging the compiled engine for inference. It can be run with both 7.x and 8.x versions of TensorRT for comparison of output.

Also in this repo are logs from runs of the script using two versions of TensorRT, built from the NVIDIA NGC containers (20.11 for 7.2.1.6 and 21.10 for 8.0.3.4). The 7.2.1.6 logs are split into two components: those from Python stdout/stderr (ending in `.txt`) and those from TensorRT at the C level (ending it `.txt.trt`. Both of these components are captured in the same file for the 8.0.3.4 log by using TensorRT's more recent custom logging funcionality.

The more recent build fails because the network stops building after the first deconvolution layer, raising the error

```
[network.cpp::setWeightsName::3013] Error Code 1: Internal Error (The given weights is not used in the network!)
```

This causes the output shape of the network to be incorrect which raises a `RuntimeError`.


## Reproducing
To run the script for a given version of the NVIDIA NGC TensorRT container, build the container with

```
TAG=20.11  # or whatever you want to use
docker build -t trt-repro:$TAG --build-arg TRT_TAG=${TAG} .
```

Then run the container with

```
docker run --rm -it -v $PWD:/workspace trt-repro:${TAG} python main.py
```

If you're leveraging a version of TensorRT earlier than 8.x and want to capture TRT's stdout, you'll have to pipe manually:

```
docker run --rm -it -v $PWD:/workspace trt-repro:${TAG} python main.py > log-${TAG}.txt.trt 2>&1
```
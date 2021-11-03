from contextlib import ExitStack
from io import BytesIO
from packaging import Version
from typing import Tuple

import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
import torch
from pycuda import autoinit  # noqa

from log_utils import TRTLogger, logger

IS_8_OR_HIGHER = Version(trt.__version) >= Version("8.0")


class OneDConvolutionalAutoencoder(torch.nn.Module):
    """Symmetric 1D Convolutional autoencoder model"""

    def __init__(self, num_channels: int) -> None:
        super().__init__()

        encoder_layers = []
        last_channels = num_channels
        for i, channels in enumerate([32, 16, 8]):
            conv = torch.nn.Conv1d(
                last_channels,
                channels,
                kernel_size=7,
                stride=2,
                padding=3
            )
            nonlin = torch.nn.ReLU()
            encoder_layers.extend([conv, nonlin])
            last_channels = channels
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i, channels in enumerate([16, 32, num_channels]):
            deconv = torch.nn.ConvTranspose1d(
                last_channels,
                channels,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1 if i > 0 else 0
            )
            nonlin = torch.nn.ReLU()
            decoder_layers.extend([deconv, nonlin])
            last_channels = channels
        self.decoder = torch.nn.Sequential(*decoder_layers[:-1])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_onnx_network(
    dummy_input: torch.Tensor
) -> Tuple[bytes, torch.Tensor]:
    """Create a 1D autoencoder and export to an ONNX binary"""
    # create the network
    nn = OneDConvolutionalAutoencoder(dummy_input.shape[1]).cuda()

    # ensure that the output is of the same shape
    # as the input i.e. we have an autoencoder
    y = nn(dummy_input)
    assert y.shape == dummy_input.shape

    # export the onnx binary to a file-like object
    logger.info("Exporting NN to ONNX")
    onnx_binary = BytesIO()
    torch.onnx.export(
        nn,
        dummy_input,
        onnx_binary,
        verbose=True,
        input_names=["x"],
        output_names=["y"]
    )

    # return the output value so that we can
    # compare what output TensorRT generates
    return onnx_binary.getvalue(), y


def convert_to_trt(onnx_binary: bytes, input_shape: Tuple[int, int, int]):
    """Convert an ONNX binary to a TensorRT executable engine"""
    logger.info("Converting ONNX model to TensorRT")
    trt_logger = TRTLogger()

    # use an exit stack to keep the number of tabs manageable
    with ExitStack() as stack:
        builder = stack.enter_context(trt.Builder(trt_logger))

        # set up the builder and config differently
        # depending on which version of TensorRT
        # we're working with
        builder.max_batch_size = input_shape[0]
        max_workspace_size = 1 << 20
        if not IS_8_OR_HIGHER:
            builder.max_workspace_size = max_workspace_size

        config = stack.enter_context(builder.create_builder_config())
        if IS_8_OR_HIGHER:
            config.max_workspace_size = max_workspace_size

        # instantiate a network and populate it
        # with the ops from our ONNX binary
        network = stack.enter_context(
            builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
        )
        parser = stack.enter_context(trt.OnnxParser(network, trt_logger))
        parser.parse(onnx_binary.getvalue())

        # if none of the layers got marked as output,
        # mark the last one as the output manually
        if network.num_outputs == 0:
            last_layer = network.get_layer(network.num_layers - 1)
            network_output = last_layer.get_output(0)
            network.mark_output(network_output)

        # make sure that the output of the network
        # matches the expected shape of our data
        output_shape = network.get_output(0).shape
        if output_shape != input_shape:
            raise RuntimeError(
                "Network output shape {} not equal to "
                "expected output shape {}".format(
                    output_shape, input_shape
                )
            )

        # build syntax is different for newer TRT
        if not IS_8_OR_HIGHER:
            # in < 8.0, build engine directly
            engine = builder.build_cuda_engine(network)
            if engine is None:
                raise RuntimeError("Build failed!")
        else:
            # in >= 8.0, build serialized binary representation
            # of engine then deserialize with Runtime
            trt_binary = builder.build_serialized_network(network, config)
            if trt_binary is None:
                raise RuntimeError("Build failed!")
            runtime = stack.enter_context(trt.Runtime(trt_logger))
            engine = runtime.deserialize_cuda_engine(trt_binary)

    return engine


def do_inference(engine, input: np.ndarray) -> np.ndarray:
    """Use a TensorRT engine to perform inference on a test input"""
    with engine, engine.create_execution_context() as context:
        stream = cuda.Stream()
        host_mems, device_mems, bindings = [], [], []

        # for the network input and output, build host
        # and device-side memory buffers for executing
        # inference on and putting results in
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            host_mems.append(host_mem)
            device_mems.append(device_mem)
            bindings.append(int(device_mem))

        logger.info("Executing inference using TensorRT engine")

        # copy the input data to the input host memory buffer
        np.copyto(host_mems[0], input)

        # copy host memory buffer to device memory buffer
        cuda.memcpy_htod_async(device_mems[0], host_mems[0], stream)

        # execute inference on device
        context.execute_async_v2(bindings, stream_handle=stream.handle)

        # copy output device memory buffer to host
        cuda.memcpy_dtoh_async(host_mems[1], device_mems[1], stream)

        # synchronize the execution stream
        stream.synchronize()

        # return the output host memory buffer data
        return host_mems[1]


def main(
    batch_size: int,
    num_channels: int,
    frame_length: int
) -> None:
    input_shape = (batch_size, num_channels, frame_length)

    # build a dummy input then build a network
    # using it. Get a target output to compare
    # against TensorRT's output
    x = torch.randn(input_shape).cuda()
    onnx_binary, y = build_onnx_network(x)

    # move these torch tensors to the numpy
    # on the CPU and flatten them for use
    # with TensorRT memory buffers
    x = x.detach().cpu().numpy().ravel()
    y = y.detach().cpu().numpy().ravel()

    # build a TensorRT engine and do some inference
    # on our dummy input using it
    engine = convert_to_trt(onnx_binary, input_shape)
    y_trt = do_inference(engine, x)

    # make sure that TensorRT's answer is pretty
    # close to Torch's answer
    err = np.abs(y_trt - y)
    assert np.isclose(err, 0, atol=1e-9).all()


if __name__ == "__main__":
    NUM_CHANNELS = 2
    FRAME_LENGTH = 100
    BATCH_SIZE = 8
    main(BATCH_SIZE, NUM_CHANNELS, FRAME_LENGTH)

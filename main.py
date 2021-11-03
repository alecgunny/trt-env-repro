import logging
import sys
from contextlib import ExitStack
from io import BytesIO
from packaging import version

import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
import torch
from pycuda import autoinit


trt_version = trt.__version__
logging.basicConfig(
    filename=f"log-{trt_version}.txt",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.debug("TensorRT version: " + trt_version)
logging.debug("System info: " + sys.platform)


class TRTLogger(trt.ILogger):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("TensorRT")

    def log(self, severity, msg):
        try:
            logger = getattr(self.logger, severity.name.lower())
        except AttributeError:
            if severity == trt.Logger.VERBOSE:
                logger = self.logger.debug
            elif severity == trt.Logger.INTERNAL_ERROR:
                logger = self.logger.error
            else:
                self.logger.exception(
                    f"Unknown severity {severity} for message {msg}"
                )
        logger(msg)


class LogCapturer:
    def __init__(self, name, level):
        self.logger = logging.getLogger(name)
        self.level = level

    def write(self, msg):
        if msg != "\n":
            self.logger.log(level=self.level, msg=msg)

    def flush(self):
        pass


sys.stdout = LogCapturer("stdout", logging.DEBUG)
sys.stderr = LogCapturer("stderr", logging.ERROR)


class OneDConvolutionalAutoencoder(torch.nn.Module):
    def __init__(self, num_channels):
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

    def __call__(self, x):
        return self.decoder(self.encoder(x))


NUM_CHANNELS = 2
FRAME_LENGTH = 100
BATCH_SIZE = 8
INPUT_SHAPE = (BATCH_SIZE, NUM_CHANNELS, FRAME_LENGTH)

nn = OneDConvolutionalAutoencoder(NUM_CHANNELS).cuda()
x = torch.randn(INPUT_SHAPE).cuda()
y = nn(x)
assert y.shape == INPUT_SHAPE

logging.info("Exporting NN to ONNX")
onnx_binary = BytesIO()
torch.onnx.export(
    nn,
    x,
    onnx_binary,
    verbose=True,
    input_names=["x"],
    output_names=["y"]
)

logging.info("Converting ONNX model to TensorRT")
logger = TRTLogger()
with ExitStack() as stack:
    builder = stack.enter_context(trt.Builder(logger))

    builder.max_batch_size = BATCH_SIZE
    max_workspace_size = 1 << 20
    if trt.__version__ < "8.0":
        builder.max_workspace_size = max_workspace_size

    config = stack.enter_context(builder.create_builder_config())
    if trt.__version__ >= "8.0":
        config.max_workspace_size = max_workspace_size

    network = stack.enter_context(
        builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
    )

    parser = stack.enter_context(trt.OnnxParser(network, logger))
    parser.parse(onnx_binary.getvalue())
    if network.num_outputs == 0:
        last_layer = network.get_layer(network.num_layers - 1)
        network_output = last_layer.get_output(0)
        network.mark_output(network_output)

    output_shape = network.get_output(0).shape
    if output_shape != INPUT_SHAPE:
        raise RuntimeError(
            "Network output shape {} not equal to "
            "expected output shape {}".format(
                output_shape, INPUT_SHAPE
            )
        )

    if trt.__version__ < "8.0":
        engine = builder.build_cuda_engine(network)
        if engine is None:
            raise RuntimeError("Build failed!")
    else:
        trt_binary = builder.build_serialized_network(network, config)
        if trt_binary is None:
            raise RuntimeError("Build failed!")
        runtime = stack.enter_context(trt.Runtime(logger))
        engine = runtime.deserialize_cuda_engine(trt_binary)

with engine, engine.create_execution_context() as context:
    stream = cuda.Stream()
    host_mems, device_mems, bindings = [], [], []
    device_mems = []
    bindings = []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
    
        host_mems.append(host_mem)
        device_mems.append(device_mem)
        bindings.append(int(device_mem))

    logging.info("Executing inference using TensorRT engine")
    np.copyto(host_mems[0], x.detach().cpu().numpy().ravel())
    cuda.memcpy_htod_async(device_mems[0], host_mems[0], stream)

    context.execute_async_v2(bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_mems[1], device_mems[1], stream)
    stream.synchronize()

    diff = np.abs(host_mems[1] - y.detach().cpu().numpy().ravel())
    assert np.isclose(diff, 0, atol=1e-9).all()


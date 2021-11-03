from contextlib import ExitStack
from io import BytesIO

import tensorrt as trt
import torch


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

nn = OneDConvolutionalAutoencoder(NUM_CHANNELS)
x = torch.randn((BATCH_SIZE, NUM_CHANNELS, FRAME_LENGTH))
y = nn(x)
assert y.shape == (BATCH_SIZE, NUM_CHANNELS, FRAME_LENGTH)

onnx_binary = BytesIO()
torch.onnx.export(
    nn,
    x,
    onnx_binary,
    verbose=True,
    input_names=["x"],
    output_names=["y"]
)

with ExitStack() as stack:
    logger = trt.Logger()
    builder = stack.enter_context(trt.Builder(logger))

    builder.max_workspace_size = 1 << 28
    builder.max_batch_size = BATCH_SIZE
    config = stack.enter_context(builder.create_builder_config())

    network = stack.enter_context(
        builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
    )
    parser = stack.enter_context(trt.OnnxParser(network, logger))
    parser.parse(onnx_binary.getvalue())
    engine = builder.build_cuda_engine(network)

buffers = [x.data_ptr(), y.data_ptr()]
stream_ptr = torch.cuda.Stream().cuda_stream

context = engine.create_execution_context()
context.execute_v2(buffers, stream_ptr)

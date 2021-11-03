import logging
import platform
import sys
from packaging.version import Version

import tensorrt as trt


trt_version = trt.__version__
logging.basicConfig(
    filename=f"log-{trt_version}.txt",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

logger.debug("TensorRT version: " + trt_version)
logger.debug("Python version: " + sys.version)
logger.debug("Platform: " + platform.platform())

IS_8_OR_HIGHER = Version(trt_version) >= Version("8.0")


class TRTLogger(trt.ILogger):
    """Custom TRT logging class for outputting to log file"""

    def __init__(self):
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
    """Wrapper class to capture stdout and stderr in log file"""

    def __init__(self, name, level):
        self.logger = logging.getLogger(name)
        self.level = level

    def write(self, msg):
        if msg != "\n":
            self.logger.log(level=self.level, msg=msg)

    def flush(self):
        pass


# catch stdout and stderr in order to log ONNX messages
sys.stdout = LogCapturer("stdout", logging.DEBUG)
sys.stderr = LogCapturer("stderr", logging.ERROR)

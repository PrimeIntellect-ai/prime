from torch.utils import cpp_extension
from pathlib import Path

INCLUDES = [str(Path(__file__).parent / "csrc")]
COMM_CSRC_PATH = Path(__file__).parent / "csrc" / "communicator.cpp"

collectives_ops = cpp_extension.load(
    name="communicator",
    sources=[COMM_CSRC_PATH],
    extra_cflags=["-O2"],
    verbose=True,
    extra_include_paths=INCLUDES,
)

SocketCommunicator = collectives_ops.SocketCommunicator

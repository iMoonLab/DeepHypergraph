from typing import Union
import sys
import logging
from pathlib import Path


def default_log_formatter() -> logging.Formatter:
    r"""Create a default formatter of log messages for logging.
    """

    return logging.Formatter("[%(levelname)s %(asctime)s]-> %(message)s")


def simple_stdout2file(file_path: Union[str, Path]) -> None:
    r""" This function simply wraps the ``sys.stdout`` stream, and outputs messages to the ``sys.stdout`` and a specified file, simultaneously.

    Args:
        ``file_path`` (``file_path: Union[str, Path]``): The path of the file to output the messages.
    """

    class SimpleLogger:
        def __init__(self, file_path: Path):
            file_path = Path(file_path).absolute()
            assert (
                file_path.parent.exists()
            ), f"The parent directory of {file_path} does not exist."
            self.file_path = file_path
            self.terminal = sys.stdout
            self.file = open(file_path, "a")

        def write(self, message):
            self.terminal.write(message)
            self.file.write(message)
            self.flush()

        def flush(self):
            self.terminal.flush()
            self.file.flush()

    file_path = Path(file_path)
    sys.stdout = SimpleLogger(file_path)

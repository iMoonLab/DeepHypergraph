from typing import Union, Optional, List, Callable
from pathlib import Path

import re
import json
import pickle as pkl


def load_from_pickle(file_path: Path, keys: Optional[Union[str, List[str]]] = None, **kwargs):
    r""" Load data from a pickle file.

    Args:
        ``file_path`` (``Path``): The local path of the file.
        ``keys`` (``Union[str, List[str]]``, optional): The keys of the data. Defaults to ``None``.
    """
    if isinstance(file_path, list):
        raise ValueError("This function only support loading data from a single file.")
    with open(file_path, "rb") as f:
        data = pkl.load(f, **kwargs)
    if keys is None:
        return data
    elif isinstance(keys, str):
        return data[keys]
    else:
        return {key: data[key] for key in keys}


def load_from_json(file_path: Path, **kwargs):
    r""" Load data from a json file.

    Args:
        ``file_path`` (``Path``): The local path of the file.
    """
    with open(file_path, "r") as f:
        data = json.load(f, **kwargs)
    return data


def load_from_txt(
    file_path: Path, dtype: Union[str, Callable], sep: str = ",| |\t", ignore_header: int = 0,
):
    r""" Load data from a txt file.

    .. note::
        The separator is a regular expression of ``re`` module. Multiple separators can be separated by ``|``. More details can refer to `re.split <https://docs.python.org/3/library/re.html#re.split>`_.

    Args:
        ``file_path`` (``Path``): The local path of the file.
        ``dtype`` (``Union[str, Callable]``): The data type of the data can be either a string or a callable function.
        ``sep`` (``str``, optional): The separator of each line in the file. Defaults to ``",| |\t"``.
        ``ignore_header`` (``int``, optional): The number of lines to ignore in the header of the file. Defaults to ``0``.
    """
    cast_fun = ret_cast_fun(dtype)
    file_path = Path(file_path)
    assert file_path.exists(), f"{file_path} does not exist."
    data = []
    with open(file_path, "r") as f:
        for _ in range(ignore_header):
            f.readline()
        data = [list(map(cast_fun, re.split(sep, line.strip()))) for line in f.readlines()]
    return data


def ret_cast_fun(dtype: Union[str, Callable]):
    r""" Return the cast function of the data type. The supported data types are: ``int``, ``float``, ``str``.

    Args:
        ``dtype`` (``Union[str, Callable]``): The data type of the data can be either a string or a callable function.
    """
    if isinstance(dtype, str):
        if dtype == "int":
            return int
        elif dtype == "float":
            return float
        elif dtype == "str":
            return str
        else:
            raise ValueError("dtype must be one of 'int', 'float', 'str'.")
    else:
        return dtype

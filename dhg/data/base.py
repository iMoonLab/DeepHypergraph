from pathlib import Path
from typing import Dict, Any, List

from dhg.datapipe import compose_pipes
from dhg.utils import download_and_check
from dhg._global import DATASETS_ROOT, REMOTE_DATASETS_ROOT


class BaseData:
    r"""The Base Class of all datasets.

    ::

        self._content = {
            'item': {
                'upon': [
                    {'filename': 'part1.pkl', 'md5': 'xxxxx',},
                    {'filename': 'part2.pkl', 'md5': 'xxxxx',},
                ],
                'loader': loader_function,
                'preprocess': [datapipe1, datapipe2],
            },
            ...
        }

    """

    def __init__(self, name: str, data_root=None):
        # configure the data local/remote root
        self.name = name
        if data_root is None:
            self.data_root = DATASETS_ROOT / name
        else:
            self.data_root = Path(data_root) / name
        self.remote_root = REMOTE_DATASETS_ROOT + name + "/"
        # init
        self._content = {}
        self._raw = {}

    def __repr__(self) -> str:
        return (
            f"This is {self.name} dataset:\n"
            + "\n".join(f"  ->  {k}" for k in self.content)
            + "\nPlease try `data['name']` to get the specified data."
        )

    @property
    def content(self):
        r"""Return the content of the dataset.
        """
        return list(self._content.keys())

    def needs_to_load(self, item_name: str) -> bool:
        r"""Return whether the ``item_name`` of the dataset needs to be loaded.

        Args:
            ``item_name`` (``str``): The name of the item in the dataset.
        """
        assert item_name in self.content, f"{item_name} is not provided in the Data"
        return (
            isinstance(self._content[item_name], dict)
            and "upon" in self._content[item_name]
            and "loader" in self._content[item_name]
        )

    def __getitem__(self, key: str) -> Any:
        if self.needs_to_load(key):
            cur_cfg = self._content[key]
            if cur_cfg.get("cache", None) is None:
                # get raw data
                item = self.raw(key)
                # preprocess and cache
                pipes = cur_cfg.get("preprocess", None)
                if pipes is not None:
                    cur_cfg["cache"] = compose_pipes(*pipes)(item)
                else:
                    cur_cfg["cache"] = item
            return cur_cfg["cache"]
        else:
            return self._content[key]

    def raw(self, key: str) -> Any:
        r"""Return the ``key`` of the dataset with un-preprocessed format.
        """
        if self.needs_to_load(key):
            cur_cfg = self._content[key]
            if self._raw.get(key, None) is None:
                upon = cur_cfg["upon"]
                if len(upon) == 0:
                    return None
                self.fetch_files(cur_cfg["upon"])
                file_path_list = [self.data_root / u["filename"] for u in cur_cfg["upon"]]
                if len(file_path_list) == 1:
                    self._raw[key] = cur_cfg["loader"](file_path_list[0])
                else:
                    # here, you should implement a multi-file loader
                    self._raw[key] = cur_cfg["loader"](file_path_list)
            return self._raw[key]
        else:
            return self._content[key]

    def fetch_files(self, files: List[Dict[str, str]]):
        r"""Download and check the files if they are not exist.

        Args:
            ``files`` (``List[Dict[str, str]]``): The files to download, each element
                in the list is a dict with at lease two keys: ``filename`` and ``md5``.
                If extra key ``bk_url`` is provided, it will be used to download the
                file from the backup url.
        """
        for file in files:
            cur_filename = file["filename"]
            cur_url = file.get("bk_url", None)
            if cur_url is None:
                cur_url = self.remote_root + cur_filename
            download_and_check(cur_url, self.data_root / cur_filename, file["md5"])

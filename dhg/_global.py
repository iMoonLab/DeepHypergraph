from pathlib import Path


def get_dhg_cache_root():

    root = Path.home() / ".cache" / "dhg"
    root.mkdir(parents=True, exist_ok=True)
    return root


AUTHOR_EMAIL = "evanfeng97@gmail.com"
# global paths
CACHE_ROOT = get_dhg_cache_root()
DATASETS_ROOT = CACHE_ROOT / "datasets"
REMOTE_DATASETS_ROOT = "https://huggingface.co/datasets/iMoonLab/DHG-data/resolve/main/datasets/"
BACKUP_REMOTE_DATASETS_ROOT = "https://download.moon-lab.tech:28501/datasets/"

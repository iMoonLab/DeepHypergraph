from pathlib import Path


def get_dhg_cache_root():

    root = Path.home() / Path(".dhg/")
    root.mkdir(parents=True, exist_ok=True)
    return root


AUTHOR_EMAIL = "evanfeng97@gmail.com"
# global paths
CACHE_ROOT = get_dhg_cache_root()
DATASETS_ROOT = CACHE_ROOT / "datasets"
# REMOTE_ROOT = "https://data.deephypergraph.com/"
REMOTE_ROOT = "https://download.moon-lab.tech:28501/"
REMOTE_DATASETS_ROOT = REMOTE_ROOT + "datasets/"
# REMOTE_DATASETS_ROOT = "https://data.shrec22.moon-lab.tech:18443/DHG/datasets/"

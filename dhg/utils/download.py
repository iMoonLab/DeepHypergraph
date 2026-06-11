import hashlib
import shutil
import requests
import warnings
from pathlib import Path
from functools import wraps


def download_file(url: str, file_path: Path):
    r""" Download a file from a url.

    Args:
        ``url`` (``str``): the url of the file
        ``file_path`` (``str``): the path to the file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, verify=True)
    if r.status_code != 200:
        raise requests.HTTPError(f"{url} is not accessible.")
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def check_file(file_path: Path, md5: str):
    r""" Check if a file is valid.

    Args:
        ``file_path`` (``Path``): The local path of the file.
        ``md5`` (``str``): The md5 of the file.

    Raises:
        FileNotFoundError: Not found the file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist.")
    else:
        with open(file_path, 'rb') as f:
            data = f.read()
        cur_md5 = hashlib.md5(data).hexdigest()
        return cur_md5 == md5


def _retry(n: int, exception_type=requests.HTTPError):
    r""" A decorator for retrying a function for n times.

    Args:
        ``n`` (``int``): The number of times to retry.
    """
    def decorator(fetcher):
        @wraps(fetcher)
        def wrapper(*args, **kwargs):
            for i in range(n - 1):
                try:
                    return fetcher(*args, **kwargs)
                except exception_type as e:
                    warnings.warn(f"Retry downloading({i + 1}/{n}): {str(e)}")
                except Exception as e:
                    raise e
            return fetcher(*args, **kwargs)
            # raise FileNotFoundError
        return wrapper
    return decorator


def _try_migrate_from_legacy_cache(file_path: Path, md5: str) -> bool:
    from dhg._global import CACHE_ROOT
    LEGACY_CACHE_ROOT = Path.home() / ".dhg"

    legacy_path = LEGACY_CACHE_ROOT / file_path.relative_to(CACHE_ROOT)
    if legacy_path is None or not legacy_path.exists(): return False
    file_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(legacy_path, file_path)
    try:
        legacy_path.parent.rmdir()
    except OSError: # If the `legacy_path.parent` is not empty, `rmdir` won't remove it. This is intended.
        pass
    return True


@_retry(3)
def download_and_check(url: str, file_path: Path, md5: str):
    r""" Download a file from a url and check its integrity.

    Args:
        ``url`` (``str``): The url of the file.
        ``file_path`` (``Path``): The path to the file.
        ``md5`` (``str``): The md5 of the file.
    """
    if not file_path.exists():
        _try_migrate_from_legacy_cache(file_path, md5) # Try to migrate from old-version cache path
    if not file_path.exists():
        download_file(url, file_path)
    if not check_file(file_path, md5):
        file_path.unlink()
        raise ValueError(
            f"{file_path} is corrupted. We will delete it, and try to download it again.")
    return True

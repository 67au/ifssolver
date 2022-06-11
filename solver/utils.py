import hashlib
from pathlib import Path

from .types import PathType


def ljust_with_zero(num_str: str) -> str:
    return f'{float(num_str):.6f}'


def parse_portal_filename(image: str, lat: str, lng: str, suffix: str = 'jpg') -> str:
    return f"{ljust_with_zero(lat)}_{ljust_with_zero(lng)}_{hashlib.md5(image.encode('utf-8')).hexdigest()}.{suffix}"


def parse_cache_path(cache_dir: PathType, method: str, image_path: PathType) -> Path:
    return Path(cache_dir).joinpath(method, f"{Path(image_path).stem}.npy")

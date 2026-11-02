"""Download Foursquare NYC and Gowalla check-in files into data/raw/."""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import requests

from lpdp.config import REPO_ROOT, load_config, resolve_path

FOURSQUARE_COLUMNS = [
    "user_raw",
    "poi_raw",
    "category_id",
    "category",
    "lat",
    "lon",
    "tz_offset",
    "ts_raw",
]
GOWALLA_COLUMNS = ["user_raw", "ts_raw", "lat", "lon", "poi_raw"]


def _download(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)


def _first_working_url(urls: list[str], dest: Path) -> Path:
    errors: list[str] = []
    for url in urls:
        try:
            _download(url, dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
    raise RuntimeError("All download URLs failed:\n" + "\n".join(errors))


def _extract_nyc_from_zip(zip_path: Path, dest: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith("dataset_TSMC2014_NYC.txt")]
        if not names:
            raise FileNotFoundError("NYC file missing from zip")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(names[0]))
    return dest


def download_foursquare_nyc(raw_dir: Path, urls: list[str]) -> Path:
    """Return path to dataset_TSMC2014_NYC.txt, downloading if needed."""
    txt = raw_dir / "foursquare_nyc" / "dataset_TSMC2014_NYC.txt"
    if txt.exists():
        return txt
    nested = list((raw_dir / "foursquare_nyc").rglob("dataset_TSMC2014_NYC.txt"))
    if nested:
        txt.parent.mkdir(parents=True, exist_ok=True)
        if nested[0].resolve() != txt.resolve():
            txt.write_bytes(nested[0].read_bytes())
        return txt
    zip_path = raw_dir / "foursquare_nyc" / "dataset_tsmc2014.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        return _extract_nyc_from_zip(zip_path, txt)
    last_err: Exception | None = None
    for url in urls:
        try:
            if url.endswith(".txt"):
                _download(url, txt)
                return txt
            _download(url, zip_path)
            return _extract_nyc_from_zip(zip_path, txt)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Could not download Foursquare NYC: {last_err}")


def download_gowalla(raw_dir: Path, urls: list[str]) -> Path:
    txt = raw_dir / "gowalla" / "loc-gowalla_totalCheckins.txt"
    if txt.exists():
        return txt
    gz = raw_dir / "gowalla" / "loc-gowalla_totalCheckins.txt.gz"
    _first_working_url(urls, gz)
    with gzip.open(gz, "rb") as src, txt.open("wb") as dst:
        dst.write(src.read())
    return txt


def download_dataset(dataset: str) -> Path:
    cfg = load_config(dataset)
    raw_dir = resolve_path(cfg, "raw")
    urls = cfg.get("download", {}).get("urls", [])
    if dataset == "foursquare_nyc":
        return download_foursquare_nyc(raw_dir, urls)
    if dataset == "gowalla":
        return download_gowalla(raw_dir, urls)
    if dataset == "synthetic":
        raw_dir.joinpath("synthetic").mkdir(parents=True, exist_ok=True)
        return raw_dir / "synthetic"
    raise ValueError(f"Unknown dataset {dataset}")


def peek_bytes(path: Path, n: int = 16) -> bytes:
    return path.read_bytes()[:n]


# Silence unused import for scripts that want REPO_ROOT.
_ = (io, REPO_ROOT)

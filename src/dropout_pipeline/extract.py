"""Download and safely extract the official UCI dataset."""

from __future__ import annotations

import hashlib
import io
import logging
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

from .exceptions import DownloadError, InvalidArchiveError

LOGGER = logging.getLogger(__name__)
UCI_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
P = ParamSpec("P")
R = TypeVar("R")


def bounded_retry(
    attempts: int = 3, delay: float = 1.0
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a function a small fixed number of times, preserving metadata."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, attempts + 1):
                try:
                    return function(*args, **kwargs)
                except (OSError, urllib.error.URLError):
                    if attempt == attempts:
                        raise
                    LOGGER.warning("Transient error; retry %d/%d", attempt + 1, attempts)
                    time.sleep(delay * attempt)
            raise RuntimeError("unreachable retry state")

        return wrapped

    return decorate


@bounded_retry()
def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "dropout-pipeline/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return bytes(response.read())


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(destination: Path, url: str = UCI_URL) -> Path:
    """Fetch and extract UCI's semicolon-delimited data.csv into destination."""
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "uci_697.zip"
    try:
        payload = _fetch(url)
    except (OSError, urllib.error.URLError) as exc:
        raise DownloadError(f"Could not download UCI dataset from {url}") from exc
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [
                member for member in archive.infolist() if Path(member.filename).name == "data.csv"
            ]
            if len(members) != 1:
                raise InvalidArchiveError("UCI archive must contain exactly one data.csv")
            data = archive.read(members[0])
    except (zipfile.BadZipFile, OSError) as exc:
        raise InvalidArchiveError("Downloaded content is not a valid ZIP archive") from exc
    archive_path.write_bytes(payload)
    csv_path = destination / "data.csv"
    csv_path.write_bytes(data)
    LOGGER.info("Extracted %s (archive SHA-256 %s)", csv_path, sha256_file(archive_path))
    return csv_path


def iter_nonempty_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Lazily yield nonblank lines with source line numbers for diagnostics."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for number, line in enumerate(stream, start=1):
            if line.strip():
                yield number, line.rstrip("\r\n")

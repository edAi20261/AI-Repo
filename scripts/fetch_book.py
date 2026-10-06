"""Fetch a book to a local file: a Google Drive link, an https URL or an s3:// URI.

    python -m scripts.fetch_book <source> <destination.pdf>

A Drive "share" link is a web page about the file, not the file, and a file
over Drive's virus-scan size answers the download with a confirmation page
instead. Both come back with status 200 — so what arrived is checked to be a
PDF before anything is spent reading it, and the error says what to fix.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from shared.config import get_config
from shared.source import FetchLimits, SourceError, materialise

_DRIVE_HOSTS = {"drive.google.com", "docs.google.com", "drive.usercontent.google.com"}
_DRIVE_FILE = re.compile(r"/file/d/([A-Za-z0-9_-]{10,})")
PDF_MAGIC = b"%PDF-"


def drive_file_id(url: str) -> str | None:
    """The file id in any form of Drive link, or ``None`` if it is not one."""
    parsed = urlparse(url)
    if parsed.hostname not in _DRIVE_HOSTS:
        return None
    match = _DRIVE_FILE.search(parsed.path)
    if match:
        return match.group(1)
    ids = parse_qs(parsed.query).get("id")
    return ids[0] if ids else None


def _download_drive(file_id: str, destination: Path, limits: FetchLimits) -> None:
    import httpx

    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}
    written = 0
    with httpx.stream("GET", url, params=params, timeout=limits.timeout_seconds,
                      follow_redirects=True) as response:
        response.raise_for_status()
        with open(destination, "wb") as sink:
            for block in response.iter_bytes(limits.chunk_bytes):
                written += len(block)
                if written > limits.max_bytes:
                    raise SourceError(f"the file is over {limits.max_bytes} bytes "
                                      "(ingestion.source.max_bytes)")
                sink.write(block)


def fetch(source: str, destination: Path) -> int:
    config = get_config()
    limits = FetchLimits.from_config(config)
    file_id = drive_file_id(source)
    if file_id:
        _download_drive(file_id, destination, limits)
    else:
        with materialise(source, config) as path:
            shutil.copyfile(path, destination)

    with open(destination, "rb") as head:
        start = head.read(1024)
    if PDF_MAGIC not in start:
        hint = (" Drive sent a web page: share the file as 'Anyone with the link' "
                "and use its link." if file_id else "")
        raise SourceError(f"what was downloaded is not a PDF.{hint}")
    return destination.stat().st_size


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    destination = Path(args[1])
    try:
        size = fetch(args[0], destination)
    except SourceError as error:
        print(f"could not fetch the book: {error}", file=sys.stderr)
        return 1
    print(f"fetched {size / 1e6:.1f} MB to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

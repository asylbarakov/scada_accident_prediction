#!/usr/bin/env python3
"""Fetch only Wind Farm A from the official CARE v6 ZIP via HTTP ranges."""

import argparse
from io import RawIOBase
from pathlib import Path, PurePosixPath
import re
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from zipfile import ZipFile
from zlib import crc32

URL = "https://zenodo.org/records/15846963/files/CARE_To_Compare.zip?download=1"
ARCHIVE_SIZE = 5_503_439_673
TRANSFER_LIMIT = 192 * 1024 * 1024
FARM = PurePosixPath("CARE_To_Compare/Wind Farm A")
METADATA_FILES = {"event_info.csv", "feature_description.csv"}


class RemoteZip(RawIOBase):
    def __init__(self):
        self.position = 0
        self.transferred = 0
        self.cache_start = 0
        self.cache = b""

    def seek(self, offset, whence=0):
        position = (0, self.position, ARCHIVE_SIZE)[whence] + offset
        if position < 0:
            raise OSError("Negative ZIP offset")
        self.position = position
        return position

    def tell(self):
        return self.position

    def read(self, size=-1):
        start = self.position
        end = ARCHIVE_SIZE if size < 0 else min(start + size, ARCHIVE_SIZE)
        if end <= start:
            return b""
        if not (self.cache_start <= start and end <= self.cache_start + len(self.cache)):
            fetch_end = min(max(end, start + 4096), ARCHIVE_SIZE)
            length = fetch_end - start
            if self.transferred + length > TRANSFER_LIMIT:
                raise OSError("Wind Farm A transfer limit reached")
            byte_range = f"{start}-{fetch_end - 1}"
            request = Request(f"{URL}&part={byte_range}", headers={
                "Range": f"bytes={byte_range}", "Accept-Encoding": "identity",
            })
            with urlopen(request, timeout=60) as response:
                if (response.status != 206 or response.headers.get("Content-Range")
                        != f"bytes {byte_range}/{ARCHIVE_SIZE}"):
                    raise OSError("Zenodo did not return the requested byte range")
                data = response.read(length + 1)
            if len(data) != length:
                raise OSError("Incomplete or oversized range response")
            self.transferred += length
            self.cache_start = start
            self.cache = data
        self.position = end
        return self.cache[start - self.cache_start:end - self.cache_start]


def fetch_farm(output: Path) -> None:
    with RemoteZip() as remote, ZipFile(remote) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()
                   and PurePosixPath(entry.filename).is_relative_to(FARM)]
        members = {PurePosixPath(entry.filename).relative_to(FARM).as_posix() for entry in entries}
        if (len(entries) != 24 or len(members) != 24 or not METADATA_FILES <= members
                or any(not re.fullmatch(r"datasets/[0-9]+\.csv", name)
                       for name in members - METADATA_FILES)):
            raise ValueError("Unexpected Wind Farm A layout in the CARE v6 ZIP")

        for entry in sorted(entries, key=lambda item: ("/datasets/" in item.filename, item.header_offset)):
            destination = output.joinpath(*PurePosixPath(entry.filename).parts)
            if destination.exists():
                with destination.open("rb") as existing:
                    checksum = 0
                    for block in iter(lambda: existing.read(1024 * 1024), b""):
                        checksum = crc32(block, checksum)
                if destination.stat().st_size != entry.file_size or checksum != entry.CRC:
                    raise ValueError(f"Existing file differs from the archive: {destination}")
                continue
            data = archive.read(entry)  # ZipFile verifies each extracted member's CRC-32.
            destination.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                staged = Path(temporary.name)
            try:
                staged.write_bytes(data)
                staged.replace(destination)
            finally:
                staged.unlink(missing_ok=True)
            print(f"Verified {entry.filename}", flush=True)
        print(f"Wind Farm A ready in {output / FARM}. HTTP bytes read: {remote.transferred:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"), help="Parent of CARE_To_Compare")
    fetch_farm(parser.parse_args().output)

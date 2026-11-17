"""Download configured source files with bounded streaming and optional SHA-256."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from .edp import sha256_file
from .preparation import write_json


def validate_file(path, suffix):
    if path.stat().st_size == 0:
        raise ValueError(f"{path} is empty")
    if suffix.lower() == ".xlsx":
        try:
            with ZipFile(path) as archive:
                archive.getinfo("xl/workbook.xml")
        except (BadZipFile, KeyError) as exc:
            raise ValueError(f"{path} is not an XLSX workbook") from exc


def acquire_file(source):
    path = Path(source["path"])
    expected = source.get("sha256")
    if path.exists():
        validate_file(path, path.suffix)
        observed = sha256_file(path)
        if expected and observed != expected:
            raise ValueError(f"SHA-256 mismatch for existing {path}")
        return observed
    if not source.get("url"):
        raise FileNotFoundError(f"Save {source['source']} as {path}, or configure its direct download URL")
    limit = int(source["max_bytes"])
    if limit <= 0:
        raise ValueError("max_bytes must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*", "Referer": source["source"]}
    staged = None
    try:
        with urlopen(Request(source["url"], headers=headers), timeout=45) as response:
            if response.headers.get_content_type() == "text/html":
                raise ValueError(f"{source['url']} returned a web page instead of a data file")
            length = response.headers.get("Content-Length")
            if length and int(length) > limit:
                raise ValueError(f"{source['url']} is larger than max_bytes")
            with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                staged = Path(handle.name)
                total = 0
                while block := response.read(min(1024 * 1024, limit - total + 1)):
                    total += len(block)
                    if total > limit:
                        raise ValueError(f"{source['url']} is larger than max_bytes")
                    handle.write(block)
        validate_file(staged, path.suffix)
        observed = sha256_file(staged)
        if expected and observed != expected:
            raise ValueError(f"SHA-256 mismatch for {source['url']}")
        staged.replace(path)
        return observed
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def acquire(config, output):
    records = []
    for source in config["signals"] + config["failure_logs"] + config.get("operation_logs", []):
        try:
            checksum = acquire_file(source)
            records.append({**source, "status": "available", "observed_sha256": checksum})
        except (OSError, ValueError) as exc:
            records.append({**source, "status": "unavailable", "error": str(exc)})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(records, output)
    return records

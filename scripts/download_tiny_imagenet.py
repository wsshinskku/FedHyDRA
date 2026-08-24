"""Download and safely extract the public Tiny-ImageNet-200 archive."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"archive member escapes destination: {member.filename}")
        bundle.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data"))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--sha256",
        dest="expected_sha256",
        help="optional expected archive digest; strongly recommended for mirrors",
    )
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.destination / "tiny-imagenet-200.zip"
    print(f"Downloading {args.url} -> {archive}")
    with urllib.request.urlopen(args.url, timeout=60) as response, archive.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    actual = sha256(archive)
    print(f"SHA256: {actual}")
    if args.expected_sha256 and actual.lower() != args.expected_sha256.lower():
        raise ValueError("downloaded archive SHA256 does not match --sha256")
    safe_extract(archive, args.destination)
    if not args.keep_archive:
        archive.unlink()
    print(f"Extracted to {args.destination / 'tiny-imagenet-200'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


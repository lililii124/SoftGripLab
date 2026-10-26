"""Portable hashes for solver records checked out on Windows and Unix."""
import hashlib
from pathlib import Path


def text_sha256(path):
    """Hash text bytes after canonicalizing CRLF to LF, as Git does for CSV."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()

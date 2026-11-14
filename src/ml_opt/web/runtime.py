"""Identify the source loaded by a running local server."""

import hashlib
from pathlib import Path


def source_revision(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    # Separate worktrees can have identical code but different local datasets.
    digest.update(str(root.resolve()).encode())
    digest.update(b"\0")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".html", ".js", ".css"):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()

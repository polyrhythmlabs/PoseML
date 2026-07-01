"""Download the official BlazePose .tflite models into models/tflite/.

Uses only the stdlib for the download itself (no extra deps). Computes and prints the
SHA-256 of each file so the hashes can be pinned into models_manifest.py for reproducible,
integrity-checked re-downloads.

Usage:
    uv run python -m poseml.convert.download_models            # download missing
    uv run python -m poseml.convert.download_models --force    # re-download all
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from poseml.models_manifest import MODELS, TFLITE_DIR, ModelSpec

DEST_DIR = TFLITE_DIR


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(spec: ModelSpec, dest: Path, force: bool) -> tuple[str, str]:
    """Returns (status, sha256)."""
    if dest.exists() and not force:
        digest = _sha256(dest)
        return ("exists", digest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  downloading {spec.url}")
    urllib.request.urlretrieve(spec.url, tmp)  # noqa: S310 (trusted Google bucket)
    digest = _sha256(tmp)

    if spec.sha256 is not None and digest != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"  checksum MISMATCH for {spec.name}\n"
            f"    expected {spec.sha256}\n    got      {digest}"
        )

    tmp.replace(dest)
    return ("downloaded", digest)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download BlazePose .tflite models.")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    print(f"Destination: {DEST_DIR}")
    results: list[tuple[str, ModelSpec, str, str]] = []
    for key, spec in MODELS.items():
        print(f"[{key}] {spec.role}")
        try:
            status, digest = _download(spec, DEST_DIR / spec.name, args.force)
        except Exception as e:  # noqa: BLE001 - report and continue to next model
            print(f"  ERROR: {e}", file=sys.stderr)
            results.append(("error", spec, str(e), ""))
            continue
        pinned = (
            "pinned-ok"
            if spec.sha256 == digest
            else ("UNPINNED" if spec.sha256 is None else "PIN-DIFFERS")
        )
        print(f"  {status}: {digest}  [{pinned}]")
        results.append((status, spec, digest, pinned))

    print("\n=== summary ===")
    for status, spec, digest, *_ in results:
        print(f"  {status:11} {spec.name:28} {digest[:16] if digest else ''}")

    unpinned = [(spec.name, digest) for status, spec, digest, pinned in results
                if pinned == "UNPINNED" and digest]
    if unpinned:
        print("\nTo pin these in models_manifest.py, set sha256= to:")
        for name, digest in unpinned:
            print(f'  {name}: "{digest}"')

    return 1 if any(r[0] == "error" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

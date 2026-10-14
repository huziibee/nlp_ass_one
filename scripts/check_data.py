"""Verify the upstream Omniglot features before analysis (no training)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external" / "icl-dynamics" / "omniglot_resnet18_randomized_order_s0.h5"
SIZE = 16_638_016
BLOB_SHA1 = "9eb588c516b641f5f25ac22227865a7ce5ab0b2f"
KEY = "resnet18/224/feat"


def inspect(path: Path = DATA, check_hdf: bool = True) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset missing: {path}\n"
            "Run: git submodule update --init --recursive\n"
            "Alternatively point --file to the official HDF5."
        )
    size = path.stat().st_size
    if size != SIZE:
        raise ValueError(f"Byte-length mismatch: {size} != {SIZE}")
    digest = hashlib.sha1()
    digest.update(f"blob {size}\\0".replace("\\0", "\0").encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    blob_sha1 = digest.hexdigest()
    if blob_sha1 != BLOB_SHA1:
        raise ValueError(f"Git blob SHA-1 differs: {blob_sha1} != {BLOB_SHA1}")
    data = {
        "path": str(path.resolve()),
        "bytes": size,
        "git_blob_sha1": blob_sha1,
        "source": "aadityasingh/icl-dynamics",
        "upstream_commit": "85b895c720844e795734b9391b32d2619065f9a1",
    }
    if check_hdf:
        import h5py
        import numpy as np
        with h5py.File(path, "r") as h:
            if KEY not in h:
                raise KeyError(f"Missing required array: {KEY}")
            arr = h[KEY]
            if arr.ndim != 3 or arr.shape[0] != 1623 or arr.shape[1] != 5:
                raise ValueError(f"Unexpected [class, exemplar, feature] shape: {arr.shape}")
            if not np.isfinite(arr[:]).all():
                raise ValueError("Nonfinite feature coordinates")
            data.update(hdf5_key=KEY, shape=list(arr.shape), dtype=str(arr.dtype))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(inspect(args.file), indent=2))


if __name__ == "__main__":
    main()

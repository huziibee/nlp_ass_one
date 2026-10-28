from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPOSITORY = "huziibee/nlp_ass_one"
ASSETS = {
    "extended": {
        "run_id": 38630071147,
        "artifact": "extended-omniglot-lr3e4-300k",
        "root": "outputs/extended_g0_full",
        "files": ("metrics.json", "extended_g0_model.eqx"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_asset(kind: str, directory: Path) -> dict:
    cfg = ASSETS[kind]
    for relative in cfg["files"]:
        item = directory / relative
        if not item.is_file() or item.stat().st_size < 32:
            raise ValueError(f"{kind}: missing or empty original file {item}")
        if not item.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"{kind}: file escapes checkpoint directory")
    meta = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if (
        meta.get("final_test_evaluated") is not False
        or meta.get("class_split") != [50, 1473, 100]
        or meta.get("steps", 0) * meta.get("batch_size", 0) != 300000
    ):
        raise ValueError("Extended G0 final-test flag, split or budget mismatch")
    return {
        "source": f"https://github.com/{REPOSITORY}/actions/runs/{cfg['run_id']}",
        "run_id": cfg["run_id"],
        "artifact": cfg["artifact"],
        "original_metadata_checked": True,
        "files": {
            relative: {
                "bytes": (directory / relative).stat().st_size,
                "sha256": sha256(directory / relative),
            }
            for relative in cfg["files"]
        },
    }


def verify(root: Path, *, kinds: tuple[str, ...] = ("extended",)) -> dict:
    return {
        "schema_version": 1,
        "checksum_scope": "downloaded local bytes",
        "independent_published_checksum_validation": False,
        "original_final_test_accessed_by_this_tool": False,
        "assets": {
            kind: check_asset(kind, root / ASSETS[kind]["root"]) for kind in kinds
        },
    }


def restore(root: Path, kinds: tuple[str, ...]) -> dict:
    exe = shutil.which("gh")
    if exe is None:
        raise RuntimeError("GitHub CLI not found. Install gh and run 'gh auth login'.")
    for kind in kinds:
        cfg = ASSETS[kind]
        target = root / cfg["root"]
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing checkpoint folder: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"nlp-{kind}-", dir=target.parent
        ) as tmp:
            staging = Path(tmp) / "unpacked"
            args = [
                exe,
                "run",
                "download",
                str(cfg["run_id"]),
                "-R",
                REPOSITORY,
                "-n",
                cfg["artifact"],
                "-D",
                str(staging),
            ]
            subprocess.run(args, check=True, timeout=600)
            check_asset(kind, staging)
            staging.rename(target)
    return verify(root, kinds=kinds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("restore", "verify"))
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--kind", choices=("extended",), default="extended")
    ap.add_argument(
        "--out", type=Path, default=REPO / "outputs/reports/CHECKPOINT_LOCK.json"
    )
    args = ap.parse_args()
    kinds = (args.kind,)
    result = (
        restore(args.root, kinds)
        if args.command == "restore"
        else verify(args.root, kinds=kinds)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"CHECKPOINTS_VERIFIED={','.join(kinds)} sha256 lock saved at {args.out}")


if __name__ == "__main__":
    main()

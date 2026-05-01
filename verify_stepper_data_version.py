from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Verify stepper checkpoint data source/version")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to s_step_*.pt")
    p.add_argument("--features", type=Path, default=root / "features_xyz_kinematic.npz")
    p.add_argument("--meta", type=Path, default=root / "features_xy_kinematic_meta.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = args.checkpoint.resolve()
    features_path = args.features.resolve()
    meta_path = args.meta.resolve()

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    fp = ckpt.get("data_fingerprint")
    if fp is None:
        print("[VERIFY] Checkpoint has no data_fingerprint (old format).")
        print("[VERIFY] Fallback checks: args/features path, dims, and clip ranges.")

    arr = np.load(features_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    x = arr["X"]
    z = arr["Z"]

    cur = {
        "features_path": str(features_path),
        "features_sha256": file_sha256(features_path),
        "meta_path": str(meta_path),
        "meta_sha256": file_sha256(meta_path),
        "x_shape": list(x.shape),
        "z_shape": list(z.shape),
        "clip_ranges_len": int(len(meta.get("clip_ranges", []))),
        "clip_ranges_head": meta.get("clip_ranges", [])[:5],
    }

    mismatches = []
    if fp is not None:
        for k in [
            "features_sha256",
            "meta_sha256",
            "x_shape",
            "z_shape",
            "clip_ranges_len",
            "clip_ranges_head",
        ]:
            if fp.get(k) != cur.get(k):
                mismatches.append(k)
        print("[VERIFY] checkpoint:", ckpt_path)
        print("[VERIFY] features:", features_path)
        print("[VERIFY] meta:", meta_path)
        if mismatches:
            print("[VERIFY] RESULT: MISMATCH")
            print("[VERIFY] mismatch keys:", ", ".join(mismatches))
        else:
            print("[VERIFY] RESULT: MATCH (same source/version)")
    else:
        ckpt_args = ckpt.get("args", {})
        print("[VERIFY] checkpoint:", ckpt_path)
        print("[VERIFY] ckpt args.features:", ckpt_args.get("features"))
        print("[VERIFY] ckpt args.meta:", ckpt_args.get("meta"))
        print("[VERIFY] current x/z dims:", x.shape[1], z.shape[1])
        print("[VERIFY] ckpt x/z dims:", ckpt.get("x_dim"), ckpt.get("z_dim"))
        if int(ckpt.get("x_dim", -1)) != int(x.shape[1]) or int(ckpt.get("z_dim", -1)) != int(z.shape[1]):
            print("[VERIFY] RESULT: MISMATCH (dimension mismatch)")
        else:
            print("[VERIFY] RESULT: PARTIAL_MATCH (dims match, but no hash-level proof)")


if __name__ == "__main__":
    main()

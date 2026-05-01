from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lmm_cd_models import Decompressor, Stepper, Projector


ROOT_DIR = Path(__file__).resolve().parent


def _find_latest_ckpt(ckpt_dir: Path, prefix: str):
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob(f"{prefix}_step_*.pt"))
    return cands[-1] if cands else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export LMM models to ONNX")
    p.add_argument(
        "--cd-checkpoint",
        type=Path,
        default=None,
        help="Path to cd checkpoint (.pt). If omitted, use latest in checkpoints_cd/",
    )
    p.add_argument(
        "--s-checkpoint",
        type=Path,
        default=None,
        help="Path to stepper checkpoint (.pt). If omitted, use latest in checkpoints_s/",
    )
    p.add_argument(
        "--p-checkpoint",
        type=Path,
        default=None,
        help="Path to projector checkpoint (.pt). If omitted, use latest in checkpoints_p/",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "onnx_models",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
    )
    return p.parse_args()


def _load_ckpt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    cd_ckpt = args.cd_checkpoint or _find_latest_ckpt(ROOT_DIR / "checkpoints_cd", "cd")
    s_ckpt = args.s_checkpoint or _find_latest_ckpt(ROOT_DIR / "checkpoints_s", "s")
    p_ckpt = args.p_checkpoint or _find_latest_ckpt(ROOT_DIR / "checkpoints_p", "p")

    if not (cd_ckpt and s_ckpt and p_ckpt):
        raise FileNotFoundError(
            "Missing checkpoint(s). Please provide --cd-checkpoint/--s-checkpoint/--p-checkpoint "
            "or ensure checkpoints directories contain files."
        )

    cd = _load_ckpt(cd_ckpt)
    s = _load_ckpt(s_ckpt)
    p = _load_ckpt(p_ckpt)

    x_dim = int(cd["x_dim"])
    y_dim = int(cd["y_dim"])
    z_dim = int(cd["z_dim"])

    d_model = Decompressor(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim).eval()
    d_model.load_state_dict(cd["decompressor_state"])

    s_model = Stepper(x_dim=int(s["x_dim"]), z_dim=int(s["z_dim"])).eval()
    s_model.load_state_dict(s["stepper_state"])

    p_model = Projector(x_dim=int(p["x_dim"]), z_dim=int(p["z_dim"])).eval()
    p_model.load_state_dict(p["projector_state"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d_path = args.out_dir / "decompressor.onnx"
    s_path = args.out_dir / "stepper.onnx"
    p_path = args.out_dir / "projector.onnx"

    x = torch.zeros(1, x_dim, dtype=torch.float32)
    z = torch.zeros(1, z_dim, dtype=torch.float32)

    torch.onnx.export(
        d_model,
        (x, z),
        d_path.as_posix(),
        input_names=["x", "z"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "z": {0: "batch"}, "y": {0: "batch"}},
        opset_version=args.opset,
    )

    torch.onnx.export(
        s_model,
        (x, z),
        s_path.as_posix(),
        input_names=["x", "z"],
        output_names=["dx", "dz"],
        dynamic_axes={
            "x": {0: "batch"},
            "z": {0: "batch"},
            "dx": {0: "batch"},
            "dz": {0: "batch"},
        },
        opset_version=args.opset,
    )

    torch.onnx.export(
        p_model,
        (x,),
        p_path.as_posix(),
        input_names=["x_query"],
        output_names=["x_proj", "z_proj"],
        dynamic_axes={
            "x_query": {0: "batch"},
            "x_proj": {0: "batch"},
            "z_proj": {0: "batch"},
        },
        opset_version=args.opset,
    )

    print(f"Exported:\n  {d_path}\n  {s_path}\n  {p_path}")
    print(f"Dimensions: x_dim={x_dim}, y_dim={y_dim}, z_dim={z_dim}, opset={args.opset}")


if __name__ == "__main__":
    main()

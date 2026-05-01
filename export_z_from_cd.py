from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lmm_cd_models import Compressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export latent Z from trained Compressor")
    p.add_argument(
        "--features",
        type=Path,
        default=Path(r"d:\learnedMotionMatching\features_xy_kinematic.npz"),
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=Path(r"d:\learnedMotionMatching\features_xy_kinematic_meta.json"),
    )
    p.add_argument(
        "--cd-checkpoint",
        type=Path,
        required=True,
        help="checkpoint path from train_compressor_decompressor.py",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(r"d:\learnedMotionMatching\features_xyz_kinematic.npz"),
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_norm(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


def fk_batch(
    local_pos: torch.Tensor, local_quat: torch.Tensor, parents: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, joints, _ = local_pos.shape
    gpos = torch.zeros_like(local_pos)
    gquat = torch.zeros_like(local_quat)
    for j in range(joints):
        p = parents[j]
        if p < 0:
            gquat[:, j] = quat_norm(local_quat[:, j])
            gpos[:, j] = local_pos[:, j]
        else:
            gquat[:, j] = quat_norm(quat_mul(gquat[:, p], local_quat[:, j]))
            gpos[:, j] = gpos[:, p] + quat_rotate(gquat[:, p], local_pos[:, j])
    return gpos, gquat


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    arr = np.load(args.features)
    x = arr["X"].astype(np.float32)
    y = arr["Y"].astype(np.float32)

    ckpt = torch.load(args.cd_checkpoint, map_location=device)
    y_layout = ckpt["y_layout"]
    parents = ckpt["parents"]
    z_dim = int(ckpt["z_dim"])
    joints = len(parents)
    y_dim = y.shape[1]
    yq_dim = y_dim + joints * 3 + joints * 4

    cnet = Compressor(yq_dim=yq_dim, z_dim=z_dim).to(device)
    cnet.load_state_dict(ckpt["compressor_state"])
    cnet.eval()

    yt_s, yt_e = y_layout["yt_local_pos"]
    yr_s, yr_e = y_layout["yr_local_quat"]

    y_t = torch.from_numpy(y).to(device)
    zs = []
    with torch.no_grad():
        for s in range(0, y_t.shape[0], args.batch_size):
            e = min(s + args.batch_size, y_t.shape[0])
            yb = y_t[s:e]
            yt = yb[:, yt_s:yt_e].reshape(e - s, joints, 3)
            yr = quat_norm(yb[:, yr_s:yr_e].reshape(e - s, joints, 4))
            qpos, qrot = fk_batch(yt, yr, parents=parents)
            q = torch.cat([qpos.reshape(e - s, -1), qrot.reshape(e - s, -1)], dim=-1)
            z = cnet(torch.cat([yb, q], dim=-1))
            zs.append(z.cpu().numpy().astype(np.float32))

    z_all = np.concatenate(zs, axis=0)
    z_mean = z_all.mean(axis=0).astype(np.float32)
    z_std = z_all.std(axis=0).astype(np.float32)
    z_std = np.where(z_std < 1e-8, 1.0, z_std).astype(np.float32)

    np.savez_compressed(
        args.out,
        X=x,
        Y=y,
        Z=z_all,
        x_mean=arr["x_mean"].astype(np.float32),
        x_std=arr["x_std"].astype(np.float32),
        z_mean=z_mean,
        z_std=z_std,
    )
    print(f"Saved: {args.out}")
    print(f"Shapes: X={x.shape}, Y={y.shape}, Z={z_all.shape}")


if __name__ == "__main__":
    main()

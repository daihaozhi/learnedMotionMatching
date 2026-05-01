from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch is required. Please install first, e.g. `pip install torch`."
    ) from exc

from lmm_cd_models import Compressor, Decompressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LMM Compressor+Decompressor")
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
        "--save-dir",
        type=Path,
        default=Path(r"d:\learnedMotionMatching\checkpoints_cd"),
    )
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=10000)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_bvh_parents(path: Path) -> list[int]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    motion_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "MOTION")
    header = lines[:motion_idx]

    parents: list[int] = []
    stack: list[int] = []
    pending_joint = False
    endsite_depth = 0

    for raw in header:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("ROOT ") or s.startswith("JOINT "):
            pending_joint = True
            continue
        if s.startswith("End Site"):
            endsite_depth = 1
            continue
        if s == "{":
            if endsite_depth > 0:
                endsite_depth += 1
            elif pending_joint:
                parent = stack[-1] if stack else -1
                parents.append(parent)
                stack.append(len(parents) - 1)
                pending_joint = False
            continue
        if s == "}":
            if endsite_depth > 0:
                endsite_depth -= 1
            elif stack:
                stack.pop()
            continue
    return parents


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
    # local_pos: [B,J,3], local_quat: [B,J,4]
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


class PairDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, pairs: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.pairs = torch.from_numpy(pairs).long()

    def __len__(self) -> int:
        return self.pairs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i0, i1 = self.pairs[idx]
        x = torch.stack([self.x[i0], self.x[i1]], dim=0)  # [2, X]
        y = torch.stack([self.y[i0], self.y[i1]], dim=0)  # [2, Y]
        return x, y


def build_pairs(clip_ranges: list[dict[str, Any]]) -> np.ndarray:
    idx_pairs = []
    for c in clip_ranges:
        s, e = int(c["start"]), int(c["end"])
        for i in range(s, e):
            idx_pairs.append((i, i + 1))
    return np.asarray(idx_pairs, dtype=np.int64)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    data = np.load(args.features)
    x = data["X"].astype(np.float32)
    y = data["Y"].astype(np.float32)

    x_dim = x.shape[1]
    y_dim = y.shape[1]
    joints = len(meta["joints"])
    parents = parse_bvh_parents(Path(meta["files"][0]))
    if len(parents) != joints:
        raise RuntimeError(f"parents count {len(parents)} != joints {joints}")

    yl = meta["Y_layout"]
    yt_s, yt_e = yl["yt_local_pos"]
    yr_s, yr_e = yl["yr_local_quat"]

    pairs = build_pairs(meta["clip_ranges"])
    ds = PairDataset(x=x, y=y, pairs=pairs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    yq_dim = y_dim + joints * 3 + joints * 4
    cnet = Compressor(yq_dim=yq_dim, z_dim=args.z_dim).to(device)
    dnet = Decompressor(x_dim=x_dim, y_dim=y_dim, z_dim=args.z_dim).to(device)

    opt = torch.optim.RAdam(
        list(cnet.parameters()) + list(dnet.parameters()), lr=args.lr
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    it = 0
    running = {
        "total": 0.0,
        "loc": 0.0,
        "chr": 0.0,
        "lvel": 0.0,
        "cvel": 0.0,
        "z_l2": 0.0,
        "z_l1": 0.0,
        "z_vel": 0.0,
    }

    while it < args.max_steps:
        for xb, yb in dl:
            if it >= args.max_steps:
                break
            # xb/yb: [B,2,D]
            xb = xb.to(device)
            yb = yb.to(device)
            bsz = xb.shape[0]

            x_flat = xb.reshape(bsz * 2, x_dim)
            y_flat = yb.reshape(bsz * 2, y_dim)

            # Decode local pose from Y.
            yt = y_flat[:, yt_s:yt_e].reshape(bsz * 2, joints, 3)
            yr = y_flat[:, yr_s:yr_e].reshape(bsz * 2, joints, 4)
            yr = quat_norm(yr)
            qpos, qrot = fk_batch(yt, yr, parents=parents)
            q_flat = torch.cat(
                [qpos.reshape(bsz * 2, -1), qrot.reshape(bsz * 2, -1)], dim=-1
            )

            z = cnet(torch.cat([y_flat, q_flat], dim=-1))
            y_hat = dnet(x_flat, z)

            yt_hat = y_hat[:, yt_s:yt_e].reshape(bsz * 2, joints, 3)
            yr_hat = quat_norm(y_hat[:, yr_s:yr_e].reshape(bsz * 2, joints, 4))
            qpos_hat, qrot_hat = fk_batch(yt_hat, yr_hat, parents=parents)

            # Pair-wise for velocity losses.
            y_pair = y_flat.reshape(bsz, 2, y_dim)
            yhat_pair = y_hat.reshape(bsz, 2, y_dim)
            q_pair = q_flat.reshape(bsz, 2, -1)
            qhat_pair = torch.cat(
                [qpos_hat.reshape(bsz * 2, -1), qrot_hat.reshape(bsz * 2, -1)], dim=-1
            ).reshape(bsz, 2, -1)

            l_loc = torch.mean(torch.abs(y_flat - y_hat))
            l_chr = torch.mean(
                torch.abs(
                    torch.cat([qpos.reshape(bsz * 2, -1), qrot.reshape(bsz * 2, -1)], dim=-1)
                    - torch.cat(
                        [qpos_hat.reshape(bsz * 2, -1), qrot_hat.reshape(bsz * 2, -1)],
                        dim=-1,
                    )
                )
            )
            l_lvel = torch.mean(
                torch.abs((y_pair[:, 1] - y_pair[:, 0]) - (yhat_pair[:, 1] - yhat_pair[:, 0]))
            )
            l_cvel = torch.mean(
                torch.abs((q_pair[:, 1] - q_pair[:, 0]) - (qhat_pair[:, 1] - qhat_pair[:, 0]))
            )
            z_pair = z.reshape(bsz, 2, -1)
            l_z_l2 = torch.mean(z * z)
            l_z_l1 = torch.mean(torch.abs(z))
            l_z_vel = torch.mean(torch.abs(z_pair[:, 1] - z_pair[:, 0]))

            # Loss weights: keep pose losses dominant, regularization small.
            loss = (
                1.0 * l_loc
                + 1.0 * l_chr
                + 0.5 * l_lvel
                + 0.5 * l_cvel
                + 1e-4 * l_z_l2
                + 1e-5 * l_z_l1
                + 1e-4 * l_z_vel
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(cnet.parameters()) + list(dnet.parameters()), max_norm=1.0
            )
            opt.step()

            it += 1
            running["total"] += float(loss.item())
            running["loc"] += float(l_loc.item())
            running["chr"] += float(l_chr.item())
            running["lvel"] += float(l_lvel.item())
            running["cvel"] += float(l_cvel.item())
            running["z_l2"] += float(l_z_l2.item())
            running["z_l1"] += float(l_z_l1.item())
            running["z_vel"] += float(l_z_vel.item())

            if it % 1000 == 0:
                for g in opt.param_groups:
                    g["lr"] *= 0.99

            if it % args.log_every == 0:
                scale = 1.0 / args.log_every
                print(
                    f"step={it} "
                    f"loss={running['total'] * scale:.6f} "
                    f"loc={running['loc'] * scale:.6f} "
                    f"chr={running['chr'] * scale:.6f} "
                    f"lvel={running['lvel'] * scale:.6f} "
                    f"cvel={running['cvel'] * scale:.6f} "
                    f"z_l2={running['z_l2'] * scale:.6f} "
                    f"z_l1={running['z_l1'] * scale:.6f} "
                    f"z_vel={running['z_vel'] * scale:.6f}"
                )
                for k in running:
                    running[k] = 0.0

            if it % args.save_every == 0 or it == args.max_steps:
                ckpt = {
                    "step": it,
                    "args": vars(args),
                    "meta_path": str(args.meta),
                    "features_path": str(args.features),
                    "parents": parents,
                    "y_layout": yl,
                    "x_dim": x_dim,
                    "y_dim": y_dim,
                    "z_dim": args.z_dim,
                    "compressor_state": cnet.state_dict(),
                    "decompressor_state": dnet.state_dict(),
                    "optimizer_state": opt.state_dict(),
                }
                out = args.save_dir / f"cd_step_{it:07d}.pt"
                torch.save(ckpt, out)
                print(f"saved checkpoint: {out}")

    print("Training done.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lmm_cd_models import Stepper

ROOT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LMM Stepper")
    p.add_argument(
        "--features",
        type=Path,
        default=ROOT_DIR / "features_xyz_kinematic.npz",
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=ROOT_DIR / "features_xy_kinematic_meta.json",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=ROOT_DIR / "checkpoints_s",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional stepper checkpoint path to resume training.",
    )
    p.add_argument("--window", type=int, default=20, help="s in paper")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=10000)
    return p.parse_args()


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, z: np.ndarray, starts: np.ndarray, window: int):
        self.x = torch.from_numpy(x).float()
        self.z = torch.from_numpy(z).float()
        self.starts = torch.from_numpy(starts).long()
        self.window = window

    def __len__(self) -> int:
        return self.starts.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx].item())
        e = s + self.window + 1
        # shape: [window+1, dim]
        return self.x[s:e], self.z[s:e]


def build_window_starts(clip_ranges: list[dict[str, Any]], window: int) -> np.ndarray:
    out: list[int] = []
    for c in clip_ranges:
        s, e = int(c["start"]), int(c["end"])
        max_start = e - window
        if max_start < s:
            continue
        out.extend(range(s, max_start + 1))
    return np.asarray(out, dtype=np.int64)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    arr = np.load(args.features)
    x = arr["X"].astype(np.float32)
    z = arr["Z"].astype(np.float32)

    x_dim = x.shape[1]
    z_dim = z.shape[1]

    starts = build_window_starts(meta["clip_ranges"], window=args.window)
    if starts.size == 0:
        raise RuntimeError("No valid windows; check clip lengths and --window")

    ds = WindowDataset(x=x, z=z, starts=starts, window=args.window)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    snet = Stepper(x_dim=x_dim, z_dim=z_dim).to(device)
    opt = torch.optim.RAdam(snet.parameters(), lr=args.lr)
    start_step = 0

    if args.resume is not None:
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location=device)
        ckpt_x_dim = int(ckpt["x_dim"])
        ckpt_z_dim = int(ckpt["z_dim"])
        if ckpt_x_dim != x_dim or ckpt_z_dim != z_dim:
            raise RuntimeError(
                f"Resume checkpoint dims mismatch: ckpt=({ckpt_x_dim},{ckpt_z_dim}) vs data=({x_dim},{z_dim})"
            )
        snet.load_state_dict(ckpt["stepper_state"])
        if "optimizer_state" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state"])
        start_step = int(ckpt.get("step", 0))
        print(f"Resumed from: {args.resume} at step={start_step}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    it = start_step
    running = {"total": 0.0, "x_val": 0.0, "z_val": 0.0, "x_vel": 0.0, "z_vel": 0.0}

    while it < args.max_steps:
        for xb, zb in dl:
            if it >= args.max_steps:
                break

            xb = xb.to(device)  # [B,s+1,x]
            zb = zb.to(device)  # [B,s+1,z]
            bsz = xb.shape[0]
            s = args.window

            x_pred = [xb[:, 0]]
            z_pred = [zb[:, 0]]
            for i in range(1, s + 1):
                dx, dz = snet(x_pred[-1], z_pred[-1])
                x_pred.append(x_pred[-1] + dx)
                z_pred.append(z_pred[-1] + dz)

            x_pred_t = torch.stack(x_pred, dim=1)  # [B,s+1,x]
            z_pred_t = torch.stack(z_pred, dim=1)  # [B,s+1,z]

            l_x_val = torch.mean(torch.abs(x_pred_t - xb))
            l_z_val = torch.mean(torch.abs(z_pred_t - zb))
            l_x_vel = torch.mean(
                torch.abs((x_pred_t[:, 1:] - x_pred_t[:, :-1]) - (xb[:, 1:] - xb[:, :-1]))
            )
            l_z_vel = torch.mean(
                torch.abs((z_pred_t[:, 1:] - z_pred_t[:, :-1]) - (zb[:, 1:] - zb[:, :-1]))
            )

            loss = 1.0 * l_x_val + 1.0 * l_z_val + 0.5 * l_x_vel + 0.5 * l_z_vel

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snet.parameters(), max_norm=1.0)
            opt.step()

            it += 1
            running["total"] += float(loss.item())
            running["x_val"] += float(l_x_val.item())
            running["z_val"] += float(l_z_val.item())
            running["x_vel"] += float(l_x_vel.item())
            running["z_vel"] += float(l_z_vel.item())

            if it % 1000 == 0:
                for g in opt.param_groups:
                    g["lr"] *= 0.99

            if it % args.log_every == 0:
                scale = 1.0 / args.log_every
                print(
                    f"step={it} loss={running['total']*scale:.6f} "
                    f"x_val={running['x_val']*scale:.6f} z_val={running['z_val']*scale:.6f} "
                    f"x_vel={running['x_vel']*scale:.6f} z_vel={running['z_vel']*scale:.6f}"
                )
                for k in running:
                    running[k] = 0.0

            if it % args.save_every == 0 or it == args.max_steps:
                ckpt = {
                    "step": it,
                    "args": vars(args),
                    "x_dim": x_dim,
                    "z_dim": z_dim,
                    "stepper_state": snet.state_dict(),
                    "optimizer_state": opt.state_dict(),
                }
                out = args.save_dir / f"s_step_{it:07d}.pt"
                torch.save(ckpt, out)
                print(f"saved checkpoint: {out}")

    print("Stepper training done.")


if __name__ == "__main__":
    main()

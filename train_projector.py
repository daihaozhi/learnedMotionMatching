from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lmm_cd_models import Projector

ROOT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LMM Projector")
    p.add_argument(
        "--features",
        type=Path,
        default=ROOT_DIR / "features_xyz_kinematic.npz",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=ROOT_DIR / "checkpoints_p",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="global multiplier for sampled n_sigma in [0,1]",
    )
    return p.parse_args()


class XDataset(Dataset):
    def __init__(self, x: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def nearest_indices(x_query: torch.Tensor, x_db: torch.Tensor) -> torch.Tensor:
    # squared euclidean argmin using ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
    q2 = torch.sum(x_query * x_query, dim=1, keepdim=True)  # [B,1]
    d2 = torch.sum(x_db * x_db, dim=1).unsqueeze(0)  # [1,N]
    dist = q2 + d2 - 2.0 * (x_query @ x_db.t())  # [B,N]
    return torch.argmin(dist, dim=1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    arr = np.load(args.features)
    x = arr["X"].astype(np.float32)
    z = arr["Z"].astype(np.float32)
    x_dim = x.shape[1]
    z_dim = z.shape[1]

    ds = XDataset(x)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    x_db = torch.from_numpy(x).float().to(device)
    z_db = torch.from_numpy(z).float().to(device)

    pnet = Projector(x_dim=x_dim, z_dim=z_dim).to(device)
    opt = torch.optim.RAdam(pnet.parameters(), lr=args.lr)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    it = 0
    running = {"total": 0.0, "x_val": 0.0, "z_val": 0.0, "dist": 0.0}

    while it < args.max_steps:
        for xb in dl:
            if it >= args.max_steps:
                break
            xb = xb.to(device)  # [B,x]
            bsz = xb.shape[0]

            sigma = torch.rand((bsz, 1), device=device) * args.noise_scale
            noise = torch.randn_like(xb)
            x_hat = xb + sigma * noise

            k = nearest_indices(x_hat, x_db)
            x_tgt = x_db[k]
            z_tgt = z_db[k]

            x_pred, z_pred = pnet(x_hat)

            l_x_val = torch.mean(torch.abs(x_pred - x_tgt))
            l_z_val = torch.mean(torch.abs(z_pred - z_tgt))
            d_ref = torch.sum((x_hat - x_tgt) ** 2, dim=1)
            d_pred = torch.sum((x_hat - x_pred) ** 2, dim=1)
            l_dist = torch.mean(torch.abs(d_ref - d_pred))

            loss = 1.0 * l_x_val + 1.0 * l_z_val + 1.0 * l_dist

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pnet.parameters(), max_norm=1.0)
            opt.step()

            it += 1
            running["total"] += float(loss.item())
            running["x_val"] += float(l_x_val.item())
            running["z_val"] += float(l_z_val.item())
            running["dist"] += float(l_dist.item())

            if it % 1000 == 0:
                for g in opt.param_groups:
                    g["lr"] *= 0.99

            if it % args.log_every == 0:
                scale = 1.0 / args.log_every
                print(
                    f"step={it} loss={running['total']*scale:.6f} "
                    f"x_val={running['x_val']*scale:.6f} "
                    f"z_val={running['z_val']*scale:.6f} "
                    f"dist={running['dist']*scale:.6f}"
                )
                for k2 in running:
                    running[k2] = 0.0

            if it % args.save_every == 0 or it == args.max_steps:
                ckpt = {
                    "step": it,
                    "args": vars(args),
                    "x_dim": x_dim,
                    "z_dim": z_dim,
                    "projector_state": pnet.state_dict(),
                    "optimizer_state": opt.state_dict(),
                }
                out = args.save_dir / f"p_step_{it:07d}.pt"
                torch.save(ckpt, out)
                print(f"saved checkpoint: {out}")

    print("Projector training done.")


if __name__ == "__main__":
    main()

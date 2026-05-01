from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def build_pairs(clip_ranges: list[dict]) -> np.ndarray:
    pairs = []
    for c in clip_ranges:
        s = int(c["start"])
        e = int(c["end"])
        if e > s:
            pairs.append(
                np.stack(
                    [np.arange(s, e, dtype=np.int64), np.arange(s + 1, e + 1, dtype=np.int64)],
                    axis=1,
                )
            )
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(pairs, axis=0)


def main() -> None:
    root = Path(__file__).resolve().parent
    meta = json.loads((root / "features_xy_kinematic_meta.json").read_text(encoding="utf-8"))
    arr = np.load(root / "features_xyz_kinematic.npz")

    x = arr["X"].astype(np.float32)
    y = arr["Y"].astype(np.float32)
    z = arr["Z"].astype(np.float32)

    yl = meta["Y_layout"]
    yt_s, _ = yl["yt_local_pos"]
    rv_s, rv_e = yl["root_vel_local"]
    ra_s, ra_e = yl["root_ang_vel_local"]

    onnx_dir = root / "onnx_models"
    p_sess = ort.InferenceSession(
        (onnx_dir / "projector.onnx").as_posix(), providers=["CPUExecutionProvider"]
    )
    s_sess = ort.InferenceSession(
        (onnx_dir / "stepper.onnx").as_posix(), providers=["CPUExecutionProvider"]
    )
    d_sess = ort.InferenceSession(
        (onnx_dir / "decompressor.onnx").as_posix(), providers=["CPUExecutionProvider"]
    )

    # A) D oracle reconstruction: D(X, Z) -> Y
    n = min(20000, x.shape[0])
    idx = np.linspace(0, x.shape[0] - 1, n, dtype=np.int64)
    xb = x[idx]
    yb = y[idx]
    zb = z[idx]
    y_hat = d_sess.run(None, {"x": xb, "z": zb})[0]

    d_mae = float(np.mean(np.abs(y_hat - yb)))
    d_rmse = float(np.sqrt(np.mean((y_hat - yb) ** 2)))
    d_root_pos_mae = float(np.mean(np.abs(y_hat[:, yt_s : yt_s + 3] - yb[:, yt_s : yt_s + 3])))
    d_root_vel_mae = float(np.mean(np.abs(y_hat[:, rv_s:rv_e] - yb[:, rv_s:rv_e])))
    d_root_ang_vel_mae = float(np.mean(np.abs(y_hat[:, ra_s:ra_e] - yb[:, ra_s:ra_e])))

    # B) P in-distribution projection: P(X) should not drift too far from X
    x_proj, z_proj = p_sess.run(None, {"x_query": xb})
    p_x_mae = float(np.mean(np.abs(x_proj - xb)))
    p_z_std = float(np.std(z_proj))

    # C) S one-step: X_t, Z_t -> X_{t+1}, Z_{t+1}
    pairs = build_pairs(meta["clip_ranges"])
    m = min(30000, pairs.shape[0])
    pidx = np.linspace(0, pairs.shape[0] - 1, m, dtype=np.int64)
    pp = pairs[pidx]
    x0 = x[pp[:, 0]]
    z0 = z[pp[:, 0]]
    x1 = x[pp[:, 1]]
    z1 = z[pp[:, 1]]

    dx_pred, dz_pred = s_sess.run(None, {"x": x0, "z": z0})
    x1_pred = x0 + dx_pred
    z1_pred = z0 + dz_pred

    s_x_mae = float(np.mean(np.abs(x1_pred - x1)))
    s_z_mae = float(np.mean(np.abs(z1_pred - z1)))

    # D) Chain one-step to pose: D(X_{t+1}^pred, Z_{t+1}^pred) vs Y_{t+1}
    y1 = y[pp[:, 1]]
    y1_pred = d_sess.run(None, {"x": x1_pred, "z": z1_pred})[0]
    chain_y_mae = float(np.mean(np.abs(y1_pred - y1)))
    chain_root_pos_mae = float(
        np.mean(np.abs(y1_pred[:, yt_s : yt_s + 3] - y1[:, yt_s : yt_s + 3]))
    )

    print("[EVAL] D_oracle_mae:", d_mae)
    print("[EVAL] D_oracle_rmse:", d_rmse)
    print("[EVAL] D_oracle_root_pos_mae:", d_root_pos_mae)
    print("[EVAL] D_oracle_root_vel_mae:", d_root_vel_mae)
    print("[EVAL] D_oracle_root_angvel_mae:", d_root_ang_vel_mae)
    print("[EVAL] P_in_dist_x_mae:", p_x_mae)
    print("[EVAL] P_in_dist_z_std:", p_z_std)
    print("[EVAL] S_1step_x_mae:", s_x_mae)
    print("[EVAL] S_1step_z_mae:", s_z_mae)
    print("[EVAL] Chain_S_then_D_Y_mae:", chain_y_mae)
    print("[EVAL] Chain_S_then_D_root_pos_mae:", chain_root_pos_mae)

    # Simple heuristic diagnosis.
    if d_root_pos_mae > 0.5:
        print("[DIAG] D is severely off even with oracle (X,Z) -> likely CD/decompressor issue.")
    elif s_x_mae > 0.15 or s_z_mae > 0.15:
        print("[DIAG] S one-step error is high -> temporal model likely unstable.")
    elif p_x_mae > 0.20:
        print("[DIAG] P projection drifts far from X -> projector likely problematic.")
    else:
        print("[DIAG] Component-wise errors are moderate; runtime pipeline mismatch is more likely.")


if __name__ == "__main__":
    main()

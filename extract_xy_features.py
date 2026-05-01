from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Any

import numpy as np


DATA_DIR = Path(r"d:\learnedMotionMatching\motion_material\kinematic_motion_normalized")
OUT_NPZ = Path(r"d:\learnedMotionMatching\features_xy_kinematic.npz")
OUT_META = Path(r"d:\learnedMotionMatching\features_xy_kinematic_meta.json")

HORIZONS = (20, 40, 60)  # 60Hz
LEFT_FOOT_NAME = "lAnkle"
RIGHT_FOOT_NAME = "rAnkle"


@dataclass
class JointDef:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str]


@dataclass
class ClipData:
    path: Path
    joints: list[JointDef]
    frame_time: float
    motion: np.ndarray  # [F, C]


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_norm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_n = np.linalg.norm(axis)
    if axis_n < 1e-12 or abs(angle) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / axis_n
    half = angle * 0.5
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def quat_from_euler_xyz_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=np.float64), math.radians(rx))
    qy = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float64), math.radians(ry))
    qz = quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float64), math.radians(rz))
    # BVH channels are Xrotation Yrotation Zrotation.
    return quat_norm(quat_mul(quat_mul(qx, qy), qz))


def quat_log_axis_scaled(q: np.ndarray) -> np.ndarray:
    # Return axis * angle (radian), useful for angular velocity.
    q = quat_norm(q)
    w = max(-1.0, min(1.0, float(q[0])))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1e-12, 1.0 - w * w))
    if s < 1e-8 or angle < 1e-10:
        return np.zeros(3, dtype=np.float64)
    axis = q[1:] / s
    return axis * angle


def inv_yaw_rotate(xz_or_xyz: np.ndarray, yaw_rad: float) -> np.ndarray:
    c = math.cos(-yaw_rad)
    s = math.sin(-yaw_rad)
    if xz_or_xyz.shape[0] == 2:
        x, z = xz_or_xyz
        return np.array([x * c - z * s, x * s + z * c], dtype=np.float64)
    x, y, z = xz_or_xyz
    return np.array([x * c - z * s, y, x * s + z * c], dtype=np.float64)


def parse_bvh(path: Path) -> ClipData:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    motion_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "MOTION":
            motion_idx = i
            break
    if motion_idx is None:
        raise ValueError(f"{path}: missing MOTION section")

    header = lines[:motion_idx]
    joints: list[JointDef] = []
    stack: list[int] = []
    pending_joint: tuple[str, int] | None = None
    endsite_depth = 0

    for raw in header:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("ROOT "):
            pending_joint = (s.split(None, 1)[1].strip(), -1)
            continue
        if s.startswith("JOINT "):
            parent = stack[-1] if stack else -1
            pending_joint = (s.split(None, 1)[1].strip(), parent)
            continue
        if s.startswith("End Site"):
            endsite_depth = 1
            continue
        if s == "{":
            if endsite_depth > 0:
                endsite_depth += 1
            elif pending_joint is not None:
                name, parent = pending_joint
                joints.append(
                    JointDef(
                        name=name,
                        parent=parent,
                        offset=np.zeros(3, dtype=np.float64),
                        channels=[],
                    )
                )
                stack.append(len(joints) - 1)
                pending_joint = None
            continue
        if s == "}":
            if endsite_depth > 0:
                endsite_depth -= 1
            elif stack:
                stack.pop()
            continue
        if endsite_depth > 0:
            continue
        if s.startswith("OFFSET "):
            if not stack:
                continue
            parts = s.split()
            joints[stack[-1]].offset = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64
            )
            continue
        if s.startswith("CHANNELS "):
            if not stack:
                continue
            parts = s.split()
            n = int(parts[1])
            joints[stack[-1]].channels = parts[2 : 2 + n]

    frames_line = lines[motion_idx + 1].strip()
    frame_time_line = lines[motion_idx + 2].strip()
    frame_count = int(frames_line.split(":", 1)[1].strip())
    frame_time = float(frame_time_line.split(":", 1)[1].strip())

    rows = []
    for raw in lines[motion_idx + 3 :]:
        s = raw.strip()
        if not s:
            continue
        rows.append([float(v) for v in s.split()])
    motion = np.asarray(rows, dtype=np.float64)
    if motion.shape[0] != frame_count:
        raise ValueError(f"{path}: declared frames={frame_count}, actual={motion.shape[0]}")

    return ClipData(path=path, joints=joints, frame_time=frame_time, motion=motion)


def extract_channels(clip: ClipData) -> dict[str, np.ndarray]:
    F = clip.motion.shape[0]
    J = len(clip.joints)

    local_pos = np.zeros((F, J, 3), dtype=np.float64)
    local_euler = np.zeros((F, J, 3), dtype=np.float64)
    local_quat = np.zeros((F, J, 4), dtype=np.float64)

    for j, joint in enumerate(clip.joints):
        local_pos[:, j, :] = joint.offset[None, :]
        local_quat[:, j, 0] = 1.0

    col = 0
    for j, joint in enumerate(clip.joints):
        for ch in joint.channels:
            vals = clip.motion[:, col]
            if ch == "Xposition":
                local_pos[:, j, 0] = vals
            elif ch == "Yposition":
                local_pos[:, j, 1] = vals
            elif ch == "Zposition":
                local_pos[:, j, 2] = vals
            elif ch == "Xrotation":
                local_euler[:, j, 0] = vals
            elif ch == "Yrotation":
                local_euler[:, j, 1] = vals
            elif ch == "Zrotation":
                local_euler[:, j, 2] = vals
            col += 1

    for f in range(F):
        for j in range(J):
            rx, ry, rz = local_euler[f, j]
            local_quat[f, j] = quat_from_euler_xyz_deg(rx, ry, rz)

    return {
        "local_pos": local_pos,
        "local_quat": local_quat,
    }


def forward_kinematics(
    joints: list[JointDef], local_pos: np.ndarray, local_quat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    F, J, _ = local_pos.shape
    global_pos = np.zeros((F, J, 3), dtype=np.float64)
    global_quat = np.zeros((F, J, 4), dtype=np.float64)

    for f in range(F):
        for j, joint in enumerate(joints):
            p = joint.parent
            if p < 0:
                global_quat[f, j] = local_quat[f, j]
                global_pos[f, j] = local_pos[f, j]
            else:
                global_quat[f, j] = quat_norm(quat_mul(global_quat[f, p], local_quat[f, j]))
                global_pos[f, j] = global_pos[f, p] + quat_rotate(global_quat[f, p], local_pos[f, j])
    return global_pos, global_quat


def finite_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float64)
    if arr.shape[0] <= 1:
        return out
    out[:-1] = (arr[1:] - arr[:-1]) / dt
    out[-1] = out[-2]
    return out


def angular_velocity_from_quat(local_quat: np.ndarray, dt: float) -> np.ndarray:
    F, J, _ = local_quat.shape
    out = np.zeros((F, J, 3), dtype=np.float64)
    if F <= 1:
        return out
    for f in range(F - 1):
        for j in range(J):
            q0 = local_quat[f, j]
            q1 = local_quat[f + 1, j]
            dq = quat_mul(quat_conj(q0), q1)  # local delta
            out[f, j] = quat_log_axis_scaled(dq) / dt
    out[-1] = out[-2]
    return out


def extract_xy_for_clip(clip: ClipData) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dt = clip.frame_time
    F = clip.motion.shape[0]
    J = len(clip.joints)

    channel_data = extract_channels(clip)
    local_pos = channel_data["local_pos"]  # [F,J,3]
    local_quat = channel_data["local_quat"]  # [F,J,4]
    global_pos, global_quat = forward_kinematics(clip.joints, local_pos, local_quat)

    names = [j.name for j in clip.joints]
    if LEFT_FOOT_NAME not in names or RIGHT_FOOT_NAME not in names:
        raise ValueError(
            f"{clip.path}: required foot joints {LEFT_FOOT_NAME}/{RIGHT_FOOT_NAME} not found"
        )
    j_left = names.index(LEFT_FOOT_NAME)
    j_right = names.index(RIGHT_FOOT_NAME)
    root = 0

    root_pos = global_pos[:, root, :]  # [F,3]
    root_vel = finite_diff(root_pos, dt)
    foot_pos_w = np.stack([global_pos[:, j_left, :], global_pos[:, j_right, :]], axis=1)  # [F,2,3]
    foot_vel_w = finite_diff(foot_pos_w, dt)

    # Facing direction from root rotation -> world forward vector projected to XZ.
    forward_w = np.zeros((F, 3), dtype=np.float64)
    yaw = np.zeros(F, dtype=np.float64)
    for f in range(F):
        fw = quat_rotate(global_quat[f, root], np.array([0.0, 0.0, 1.0], dtype=np.float64))
        fw[1] = 0.0
        n = np.linalg.norm(fw)
        if n < 1e-8:
            fw = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            fw = fw / n
        forward_w[f] = fw
        yaw[f] = math.atan2(fw[0], fw[2])  # x,z

    # Build X: [tt(6), td(6), ft(6), ftdot(6), hdot(3)] = 27
    X = np.zeros((F, 27), dtype=np.float64)
    for i in range(F):
        yi = yaw[i]
        root_i = root_pos[i]
        x_list: list[float] = []

        # future trajectory positions (2D) and directions (2D)
        for h in HORIZONS:
            k = min(i + h, F - 1)
            rel = root_pos[k] - root_i
            rel2 = np.array([rel[0], rel[2]], dtype=np.float64)
            lp = inv_yaw_rotate(rel2, yi)
            x_list.extend([float(lp[0]), float(lp[1])])

        for h in HORIZONS:
            k = min(i + h, F - 1)
            fd = np.array([forward_w[k, 0], forward_w[k, 2]], dtype=np.float64)
            ld = inv_yaw_rotate(fd, yi)
            x_list.extend([float(ld[0]), float(ld[1])])

        # feet pos / vel local to current character heading
        for side in range(2):
            rel = foot_pos_w[i, side] - root_i
            lp = inv_yaw_rotate(rel, yi)
            x_list.extend([float(lp[0]), float(lp[1]), float(lp[2])])

        for side in range(2):
            lv = inv_yaw_rotate(foot_vel_w[i, side], yi)
            x_list.extend([float(lv[0]), float(lv[1]), float(lv[2])])

        # hip velocity (use root translational velocity in local heading frame)
        hv = inv_yaw_rotate(root_vel[i], yi)
        x_list.extend([float(hv[0]), float(hv[1]), float(hv[2])])

        X[i] = np.asarray(x_list, dtype=np.float64)

    # Build Y (training pose target, flattened)
    local_pos_vel = finite_diff(local_pos, dt)  # [F,J,3]
    local_ang_vel = angular_velocity_from_quat(local_quat, dt)  # [F,J,3]
    root_vel_local = np.zeros((F, 3), dtype=np.float64)
    for i in range(F):
        root_vel_local[i] = inv_yaw_rotate(root_vel[i], yaw[i])
    root_ang_vel = local_ang_vel[:, root, :]

    yt = local_pos.reshape(F, J * 3)
    yr = local_quat.reshape(F, J * 4)
    ydt = local_pos_vel.reshape(F, J * 3)
    ydr = local_ang_vel.reshape(F, J * 3)
    Y = np.concatenate([yt, yr, ydt, ydr, root_vel_local, root_ang_vel], axis=1).astype(np.float64)

    info = {
        "file": str(clip.path),
        "frames": F,
        "joints": J,
    }
    return X, Y, info


def main() -> None:
    files = sorted(DATA_DIR.glob("*.bvh"))
    if not files:
        raise RuntimeError(f"No bvh files found in {DATA_DIR}")

    clips: list[ClipData] = [parse_bvh(p) for p in files]
    frame_times = [c.frame_time for c in clips]
    if max(frame_times) - min(frame_times) > 1e-8:
        raise RuntimeError(f"Inconsistent frame time in dataset: {frame_times}")

    # Ensure same skeleton names/order.
    base_names = [j.name for j in clips[0].joints]
    for c in clips[1:]:
        names = [j.name for j in c.joints]
        if names != base_names:
            raise RuntimeError(f"Skeleton mismatch in {c.path}")

    all_X = []
    all_Y = []
    clip_ranges = []
    clip_infos = []
    cursor = 0
    for c in clips:
        X, Y, info = extract_xy_for_clip(c)
        all_X.append(X)
        all_Y.append(Y)
        start = cursor
        end = cursor + X.shape[0] - 1
        clip_ranges.append((start, end, c.path.name))
        cursor = end + 1
        clip_infos.append(info)

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    Y = np.concatenate(all_Y, axis=0).astype(np.float32)

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)

    np.savez_compressed(
        OUT_NPZ,
        X=X,
        Y=Y,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
    )

    J = len(base_names)
    meta = {
        "source_dir": str(DATA_DIR),
        "files": [str(p) for p in files],
        "frame_time": clips[0].frame_time,
        "total_frames": int(X.shape[0]),
        "X_dim": int(X.shape[1]),
        "Y_dim": int(Y.shape[1]),
        "joints": base_names,
        "foot_joints": [LEFT_FOOT_NAME, RIGHT_FOOT_NAME],
        "horizons": list(HORIZONS),
        "clip_ranges": [
            {"start": int(s), "end": int(e), "file": f} for (s, e, f) in clip_ranges
        ],
        "X_layout": [
            "traj_pos_t+20(x,z)",
            "traj_pos_t+40(x,z)",
            "traj_pos_t+60(x,z)",
            "traj_dir_t+20(x,z)",
            "traj_dir_t+40(x,z)",
            "traj_dir_t+60(x,z)",
            f"{LEFT_FOOT_NAME}_pos_local(x,y,z)",
            f"{RIGHT_FOOT_NAME}_pos_local(x,y,z)",
            f"{LEFT_FOOT_NAME}_vel_local(x,y,z)",
            f"{RIGHT_FOOT_NAME}_vel_local(x,y,z)",
            "hip_vel_local(x,y,z) [implemented as root vel]",
        ],
        "Y_layout": {
            "yt_local_pos": [0, J * 3],
            "yr_local_quat": [J * 3, J * 3 + J * 4],
            "ydt_local_pos_vel": [J * 3 + J * 4, J * 6 + J * 4],
            "ydr_local_ang_vel": [J * 6 + J * 4, J * 9 + J * 4],
            "root_vel_local": [J * 9 + J * 4, J * 9 + J * 4 + 3],
            "root_ang_vel_local": [J * 9 + J * 4 + 3, J * 9 + J * 4 + 6],
        },
        "notes": [
            "All trajectory/velocity features are local to current root heading (yaw-only).",
            "Future indices are clamped to clip end.",
            "Use x_std (and optional x_mean) for normalization in training/runtime.",
        ],
        "clip_infos": clip_infos,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_NPZ}")
    print(f"Saved: {OUT_META}")
    print(f"X shape={X.shape}, Y shape={Y.shape}")


if __name__ == "__main__":
    main()

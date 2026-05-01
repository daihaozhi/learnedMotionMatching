from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import statistics


DATA_DIR = Path(r"d:\learnedMotionMatching\motion_material\kinematic_motion")
OUT_DIR = Path(r"d:\learnedMotionMatching\motion_material\kinematic_motion_normalized")
REPORT_PATH = Path(r"d:\learnedMotionMatching\kinematic_preprocess_report.txt")


@dataclass
class BVHData:
    path: Path
    header_lines: list[str]
    root_channels: list[str]
    frame_count_declared: int
    frame_time: float
    motion_rows: list[list[float]]


def parse_bvh(path: Path) -> BVHData:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    motion_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "MOTION":
            motion_idx = i
            break
    if motion_idx is None:
        raise ValueError(f"{path} missing MOTION section")

    header = lines[:motion_idx]
    root_channels: list[str] = []
    seen_root = False
    root_opened = False
    brace_depth = 0
    for ln in header:
        s = ln.strip()
        if s.startswith("ROOT "):
            seen_root = True
            continue
        if seen_root and s == "{":
            root_opened = True
            brace_depth = 1
            continue
        if root_opened:
            if s == "{":
                brace_depth += 1
            elif s == "}":
                brace_depth -= 1
            elif brace_depth == 1 and s.startswith("CHANNELS "):
                parts = s.split()
                n = int(parts[1])
                root_channels = parts[2 : 2 + n]
                break

    if not root_channels:
        raise ValueError(f"{path} missing root channels")

    frames_line = lines[motion_idx + 1].strip()
    frame_time_line = lines[motion_idx + 2].strip()
    if not frames_line.startswith("Frames:"):
        raise ValueError(f"{path} invalid Frames line: {frames_line}")
    if not frame_time_line.startswith("Frame Time:"):
        raise ValueError(f"{path} invalid Frame Time line: {frame_time_line}")

    frame_count = int(frames_line.split(":", 1)[1].strip())
    frame_time = float(frame_time_line.split(":", 1)[1].strip())

    motion_rows: list[list[float]] = []
    for raw in lines[motion_idx + 3 :]:
        s = raw.strip()
        if not s:
            continue
        vals = [float(x) for x in s.split()]
        motion_rows.append(vals)

    return BVHData(
        path=path,
        header_lines=header,
        root_channels=root_channels,
        frame_count_declared=frame_count,
        frame_time=frame_time,
        motion_rows=motion_rows,
    )


def contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    out: list[tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append((start, prev))
        start = i
        prev = i
    out.append((start, prev))
    return out


def rotation_y_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> float:
    # XYZ euler (degree) to yaw around global Y (extract from rotation matrix)
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # R = Rx * Ry * Rz
    r00 = cy * cz
    r02 = sy
    yaw = math.atan2(r02, r00)
    return math.degrees(yaw)


def normalize_root(data: BVHData) -> list[list[float]]:
    rc = data.root_channels
    ix = rc.index("Xposition")
    iy = rc.index("Yposition")
    iz = rc.index("Zposition")
    irx = rc.index("Xrotation")
    iry = rc.index("Yrotation")
    irz = rc.index("Zrotation")

    first = data.motion_rows[0]
    x0, y0, z0 = first[ix], first[iy], first[iz]

    # Use initial yaw as heading baseline.
    yaw0 = rotation_y_deg(first[irx], first[iry], first[irz])
    yaw0_rad = math.radians(yaw0)
    cos_y = math.cos(-yaw0_rad)
    sin_y = math.sin(-yaw0_rad)

    out: list[list[float]] = []
    for row in data.motion_rows:
        r = list(row)
        dx = r[ix] - x0
        dy = r[iy] - y0
        dz = r[iz] - z0

        nx = dx * cos_y - dz * sin_y
        nz = dx * sin_y + dz * cos_y
        r[ix] = nx
        r[iy] = dy
        r[iz] = nz
        r[iry] = r[iry] - yaw0
        out.append(r)
    return out


def write_bvh(data: BVHData, rows: list[list[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for ln in data.header_lines:
            f.write(ln + "\n")
        f.write("MOTION\n")
        f.write(f"Frames: {len(rows)}\n")
        f.write(f"Frame Time:   {data.frame_time:.6f}\n")
        for row in rows:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def main() -> None:
    files = sorted(DATA_DIR.glob("*.bvh"))
    if not files:
        raise RuntimeError(f"No BVH found in {DATA_DIR}")

    parsed = [parse_bvh(p) for p in files]
    std = parsed[0]

    report: list[str] = []
    report.append(f"DATA_DIR: {DATA_DIR}")
    report.append(f"STANDARD_FILE: {std.path.name}")
    report.append(f"FILE_COUNT: {len(parsed)}")
    report.append("")
    report.append("== Coordinate/Schema Check ==")

    mismatch_files: list[str] = []
    for d in parsed:
        ok = True
        if d.root_channels != std.root_channels:
            ok = False
            report.append(f"[Mismatch] {d.path.name}: root_channels differ")
            report.append(f"  got: {d.root_channels}")
            report.append(f"  std: {std.root_channels}")
        if abs(d.frame_time - std.frame_time) > 1e-7:
            ok = False
            report.append(
                f"[Mismatch] {d.path.name}: frame_time {d.frame_time} != {std.frame_time}"
            )
        if not ok:
            mismatch_files.append(d.path.name)

    if not mismatch_files:
        report.append("All files have consistent root channels and frame time.")

    report.append("")
    report.append("== Root Anomaly Check ==")
    report.append(
        "Rule: mark frame if planar speed > median + 6*MAD OR > 10 m/s (hard cap)."
    )

    rc = std.root_channels
    ix = rc.index("Xposition")
    iz = rc.index("Zposition")
    for d in parsed:
        dt = d.frame_time
        speed: list[float] = []
        for i in range(1, len(d.motion_rows)):
            a, b = d.motion_rows[i - 1], d.motion_rows[i]
            vx = (b[ix] - a[ix]) / dt
            vz = (b[iz] - a[iz]) / dt
            speed.append(math.sqrt(vx * vx + vz * vz))

        med = statistics.median(speed) if speed else 0.0
        abs_dev = [abs(v - med) for v in speed]
        mad = statistics.median(abs_dev) if abs_dev else 0.0
        thr = max(med + 6.0 * mad, 10.0)
        bad = [i + 1 for i, v in enumerate(speed) if v > thr]
        ranges = contiguous_ranges(bad)
        report.append(
            f"{d.path.name}: median_speed={med:.3f} m/s, mad={mad:.3f}, threshold={thr:.3f}, anomaly_frames={len(bad)}"
        )
        if ranges:
            preview = ", ".join([f"[{s}-{e}]" for s, e in ranges[:10]])
            report.append(f"  anomaly_ranges(first10): {preview}")
        else:
            report.append("  anomaly_ranges: none")

    report.append("")
    report.append("== Root Normalization ==")
    report.append(
        "Operation: set first-frame root to origin; remove initial yaw from root translation and Y-rotation."
    )
    report.append(f"Output dir: {OUT_DIR}")

    for d in parsed:
        norm = normalize_root(d)
        out_path = OUT_DIR / d.path.name
        write_bvh(d, norm, out_path)
        report.append(f"normalized: {d.path.name} -> {out_path}")

    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")
    print(f"Normalized BVH dir: {OUT_DIR}")


if __name__ == "__main__":
    main()

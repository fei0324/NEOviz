"""
Save 3D propagated variant states at uncertainty-tube cut-plane times.

This script reads the cut-plane times from the tube JSON and samples each variant
SPICE kernel at those exact times. It saves the resulting 3D point cloud and
velocity cloud as a NumPy .npz file.

The saved arrays include:

    cutplane_indices        (num_times,)
    times_utc               (num_times,)
    times_et                (num_times,)
    target_ids              (num_variants,)
    positions_km            (num_times, num_variants, 3)
    velocities_km_s         (num_times, num_variants, 3)

By default, this saves only those arrays. Use --include-centers if you want to
embed the cut-plane centers too; otherwise they can be recovered from the tube
JSON using cutplane_indices.

By default, only the first five cut planes are sampled for a quick test. Use
--all-cutplanes to save every cut plane.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TUBE_JSON = REPO_ROOT / "data" / "B612_data" / "Test" / "2004 MN4" / "tube_2004_MN4.json"
DEFAULT_VARIANT_KERNELS = (
    REPO_ROOT / "data" / "B612_data" / "Test" / "2004 MN4" / "variant_kernels"
)
DEFAULT_OUTPUT = REPO_ROOT / "revision" / "cache" / "2004_MN4_cutplane_states.npz"

LSK_KERNEL = REPO_ROOT / "data" / "kernels" / "lsk" / "naif0012.tls.pc"
SPK_KERNEL = REPO_ROOT / "data" / "kernels" / "spk" / "de432s.bsp"
PCK_KERNEL = REPO_ROOT / "data" / "kernels" / "pck" / "pck00011.tpc"

SPICE_OFFSET = 1_000_000
SUN_ID = "10"
FRAME = "ECLIPJ2000"
ABCORR = "NONE"
METERS_PER_KM = 1000.0


@dataclass(frozen=True)
class CutPlane:
    index: int
    time_utc: str
    center_km: np.ndarray


def load_cutplanes(tube_json: Path) -> list[CutPlane]:
    with tube_json.open("r", encoding="utf-8") as file:
        tube_data = json.load(file)

    cutplanes = []
    for index, polygon in enumerate(tube_data["polygons"]):
        center = polygon["center"]
        center_km = np.array([center["x"], center["y"], center["z"]], dtype=float)
        center_km /= METERS_PER_KM
        cutplanes.append(CutPlane(index=index, time_utc=polygon["time"], center_km=center_km))

    return cutplanes


def parse_indices(raw_indices: str | None, count: int, all_cutplanes: bool) -> list[int]:
    if raw_indices:
        indices = [int(value.strip()) for value in raw_indices.split(",") if value.strip()]
    elif all_cutplanes:
        indices = list(range(count))
    else:
        indices = list(range(min(5, count)))

    return sorted(set(index for index in indices if 0 <= index < count))


def variant_kernel_paths(variant_dir: Path, max_variants: int | None) -> list[Path]:
    kernels = sorted(variant_dir.glob("*.bsp"))
    if max_variants is not None:
        kernels = kernels[:max_variants]
    if not kernels:
        raise FileNotFoundError(f"No .bsp variant kernels found in {variant_dir}")
    return kernels


def target_id_from_kernel(kernel_path: Path) -> int:
    """Map 000001.bsp -> 1000000, 000002.bsp -> 1000001, etc."""

    variant_number = int(kernel_path.stem)
    return SPICE_OFFSET + variant_number - 1


def core_kernel_paths() -> list[str]:
    kernels = [LSK_KERNEL, SPK_KERNEL, PCK_KERNEL]
    for kernel in kernels:
        if not kernel.exists():
            raise FileNotFoundError(f"Required SPICE kernel not found: {kernel}")
    return [str(kernel) for kernel in kernels]


def compute_times_et(times_utc: list[str], core_kernels: list[str]) -> np.ndarray:
    import spiceypy as spice

    try:
        for kernel in core_kernels:
            spice.furnsh(kernel)
        return np.array([spice.str2et(time_utc) for time_utc in times_utc], dtype=float)
    finally:
        spice.kclear()


def sample_variant_worker(
    variant_index: int,
    kernel_path: str,
    target_id: int,
    times_utc: list[str],
    core_kernels: list[str],
) -> tuple[int, int, str, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Worker process: load one variant BSP and sample it at every cut-plane time.

    Returning one full time series per variant avoids sharing SPICE global state
    across threads or processes.
    """

    import spiceypy as spice

    positions_km = np.full((len(times_utc), 3), np.nan)
    velocities_km_s = np.full((len(times_utc), 3), np.nan)
    missing_by_time = np.zeros(len(times_utc), dtype=np.int8)
    first_error = ""

    try:
        for kernel in core_kernels:
            spice.furnsh(kernel)
        spice.furnsh(kernel_path)

        for time_index, time_utc in enumerate(times_utc):
            try:
                et = spice.str2et(time_utc)
                state, _ = spice.spkezr(str(target_id), et, FRAME, ABCORR, SUN_ID)
            except Exception as error:
                missing_by_time[time_index] = 1
                if not first_error:
                    first_error = str(error).splitlines()[0]
                continue

            positions_km[time_index] = state[:3]
            velocities_km_s[time_index] = state[3:]

        return (
            variant_index,
            target_id,
            Path(kernel_path).name,
            positions_km,
            velocities_km_s,
            missing_by_time,
            first_error,
        )
    finally:
        spice.kclear()


def save_npz(output_path: Path, compressed: bool, **arrays) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(output_path, **arrays)
    else:
        np.savez(output_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tube-json", type=Path, default=DEFAULT_TUBE_JSON)
    parser.add_argument("--variant-kernels", type=Path, default=DEFAULT_VARIANT_KERNELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma-separated cut-plane indices to sample. Default: first five cut planes.",
    )
    parser.add_argument(
        "--all-cutplanes",
        action="store_true",
        help="Sample every cut plane from the tube JSON.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Use only the first N variant kernels. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel worker processes. Default: CPU count minus one.",
    )
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed. Larger file, faster save.",
    )
    parser.add_argument(
        "--include-centers",
        action="store_true",
        help="Also save centers_km. By default centers are omitted to keep the cache minimal.",
    )
    parser.add_argument(
        "--include-missing-counts",
        action="store_true",
        help="Also save per-variant missing state counts.",
    )
    parser.add_argument(
        "--include-units",
        action="store_true",
        help="Also save small unit/frame metadata strings.",
    )
    args = parser.parse_args()

    cutplanes = load_cutplanes(args.tube_json)
    selected_indices = parse_indices(args.indices, len(cutplanes), args.all_cutplanes)
    selected_cutplanes = [cutplanes[index] for index in selected_indices]
    variant_kernels = variant_kernel_paths(args.variant_kernels, args.max_variants)
    core_kernels = core_kernel_paths()

    times_utc = [cutplane.time_utc for cutplane in selected_cutplanes]
    centers_km = np.vstack([cutplane.center_km for cutplane in selected_cutplanes])
    cutplane_indices = np.array(selected_indices, dtype=np.int32)
    target_ids = np.array([target_id_from_kernel(kernel) for kernel in variant_kernels], dtype=np.int32)

    print(f"Loaded {len(cutplanes)} cut planes from {args.tube_json}", flush=True)
    print(f"Sampling {len(selected_cutplanes)} cut-plane times", flush=True)
    print(f"Sampling {len(variant_kernels)} variant kernels from {args.variant_kernels}", flush=True)
    print(f"Using {args.workers} worker processes", flush=True)
    print(f"Output: {args.output}", flush=True)

    print("Computing SPICE ephemeris times for selected cut planes...", flush=True)
    times_et = compute_times_et(times_utc, core_kernels)

    positions_km = np.full((len(selected_cutplanes), len(variant_kernels), 3), np.nan)
    velocities_km_s = np.full((len(selected_cutplanes), len(variant_kernels), 3), np.nan)
    missing_counts = np.zeros(len(variant_kernels), dtype=np.int64)
    missing_by_time = np.zeros(len(selected_cutplanes), dtype=np.int64)
    first_worker_error = ""

    worker_count = max(1, min(args.workers, len(variant_kernels)))
    print("Starting parallel variant sampling...", flush=True)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                sample_variant_worker,
                variant_index,
                str(kernel),
                int(target_ids[variant_index]),
                times_utc,
                core_kernels,
            )
            for variant_index, kernel in enumerate(variant_kernels)
        ]

        completed = 0
        for future in as_completed(futures):
            (
                variant_index,
                target_id,
                kernel_name,
                positions,
                velocities,
                variant_missing_by_time,
                first_error,
            ) = future.result()
            positions_km[:, variant_index, :] = positions
            velocities_km_s[:, variant_index, :] = velocities
            missing_count = int(variant_missing_by_time.sum())
            missing_counts[variant_index] = missing_count
            missing_by_time += variant_missing_by_time.astype(np.int64)
            if first_error and not first_worker_error:
                first_worker_error = first_error

            completed += 1
            if completed == 1 or completed == len(futures) or completed % 25 == 0:
                print(
                    f"  Completed {completed}/{len(futures)} variants "
                    f"(latest {kernel_name}, target {target_id}, missing {missing_count})",
                    flush=True,
                )

    total_missing = int(missing_counts.sum())
    print(f"Finished sampling. Missing state count: {total_missing}", flush=True)
    if total_missing:
        print("Missing states by cut-plane time:", flush=True)
        for cutplane, count in zip(selected_cutplanes, missing_by_time):
            if count:
                print(
                    f"  index {cutplane.index}, time {cutplane.time_utc}: "
                    f"{count}/{len(variant_kernels)} missing",
                    flush=True,
                )
        if first_worker_error:
            print(f"First SPICE error seen: {first_worker_error}", flush=True)
    print("Saving NumPy state cache...", flush=True)
    arrays_to_save = {
        "cutplane_indices": cutplane_indices,
        "times_utc": np.array(times_utc),
        "times_et": times_et,
        "target_ids": target_ids,
        "positions_km": positions_km,
        "velocities_km_s": velocities_km_s,
    }
    if args.include_centers:
        arrays_to_save["centers_km"] = centers_km
    if args.include_missing_counts:
        arrays_to_save["missing_counts"] = missing_counts
        arrays_to_save["missing_by_time"] = missing_by_time
    if args.include_units:
        arrays_to_save["position_units"] = np.array("km")
        arrays_to_save["velocity_units"] = np.array("km/s")
        arrays_to_save["frame"] = np.array(FRAME)
        arrays_to_save["observer"] = np.array("Sun")

    save_npz(args.output, not args.uncompressed, **arrays_to_save)
    print(f"Wrote {args.output}", flush=True)
    print(f"Saved arrays: {', '.join(arrays_to_save.keys())}", flush=True)


if __name__ == "__main__":
    main()

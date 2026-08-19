"""
Extract and plot propagated variant states/intersections on uncertainty-tube cut planes.

This script reads the cut-plane times and centers from the tube JSON, reads the
propagated variant SPICE kernels, recomputes where each variant intersects each
cut plane, and plots 2D density views.

Run from the repository root with:

    python revision/extract_cutplanes.py

By default, plots are shown interactively. Pass --save-plots to write PNG files to:

    revision/output/cutplanes/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

DEFAULT_TUBE_JSON = REPO_ROOT / "data" / "B612_data" / "Test" / "2004 MN4" / "tube_2004_MN4.json"
DEFAULT_VARIANT_KERNELS = (
    REPO_ROOT / "data" / "B612_data" / "Test" / "2004 MN4" / "variant_kernels"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "revision" / "output" / "cutplanes"

LSK_KERNEL = REPO_ROOT / "data" / "kernels" / "lsk" / "naif0012.tls.pc"
SPK_KERNEL = REPO_ROOT / "data" / "kernels" / "spk" / "de432s.bsp"
PCK_KERNEL = REPO_ROOT / "data" / "kernels" / "pck" / "pck00011.tpc"

SUN_ID = "10"
FRAME = "ECLIPJ2000"
ABCORR = "NONE"
METERS_PER_KM = 1000.0


@dataclass(frozen=True)
class CutPlane:
    index: int
    time_utc: str
    center_km: np.ndarray


def import_runtime_dependencies():
    """Import plotting/SPICE dependencies with a friendly error if missing."""

    try:
        import matplotlib.pyplot as plt
        import spiceypy as spice
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        import tube_generation.util as tube_util
    except ImportError as error:
        raise SystemExit(
            "Missing a required package. Install/use an environment with "
            "`spiceypy`, `numpy`, `matplotlib`, and the repo's analysis "
            "dependencies available.\n"
            f"Original import error: {error}"
        ) from error

    return plt, spice, tube_util


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


def variant_kernel_paths(variant_dir: Path, max_variants: int | None) -> list[Path]:
    kernels = sorted(variant_dir.glob("*.bsp"))
    if max_variants is not None:
        kernels = kernels[:max_variants]
    if not kernels:
        raise FileNotFoundError(f"No .bsp variant kernels found in {variant_dir}")
    return kernels


def target_id_from_kernel(tube_util, kernel_path: Path) -> int:
    """Map 000001.bsp -> 1000000, 000002.bsp -> 1000001, etc."""

    variant_number = int(kernel_path.stem)
    return tube_util.SPICE_OFFSET + variant_number - 1


def furnish_kernels(spice, variant_kernels: list[Path]) -> None:
    for kernel in [LSK_KERNEL, SPK_KERNEL, PCK_KERNEL]:
        if not kernel.exists():
            raise FileNotFoundError(f"Required SPICE kernel not found: {kernel}")
        print(f"Loading core kernel: {kernel.name}", flush=True)
        spice.furnsh(str(kernel))

    for index, kernel in enumerate(variant_kernels, start=1):
        if index == 1 or index == len(variant_kernels) or index % 50 == 0:
            print(
                f"Loading variant kernel {index}/{len(variant_kernels)}: {kernel.name}",
                flush=True,
            )
        spice.furnsh(str(kernel))


def states_at_cutplane(
    spice,
    tube_util,
    et: float,
    variant_kernels: list[Path],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    positions = []
    velocities = []
    target_ids = []

    for index, kernel in enumerate(variant_kernels, start=1):
        if index == 1 or index == len(variant_kernels) or index % 50 == 0:
            print(
                f"  Reading state {index}/{len(variant_kernels)} from {kernel.name}",
                flush=True,
            )
        target_id = target_id_from_kernel(tube_util, kernel)
        try:
            state, _ = spice.spkezr(str(target_id), et, FRAME, ABCORR, SUN_ID)
        except Exception as error:
            print(f"Skipping {kernel.name} / target {target_id}: {error}")
            continue

        positions.append(np.array(state[:3], dtype=float))
        velocities.append(np.array(state[3:], dtype=float))
        target_ids.append(target_id)

    if not positions:
        raise RuntimeError("No variant states were available for this cut-plane time.")

    return np.vstack(positions), np.vstack(velocities), target_ids


def intersect_variants_with_plane(
    tube_util,
    center_km: np.ndarray,
    positions_km: np.ndarray,
    velocities_km_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Intersect each variant's local position/velocity line with the cut plane.

    The normal follows the original tube-generation idea: use the median variant
    velocity at the cut-plane time. Because velocities are in km/s, the returned
    time offsets are seconds.
    """

    normal = np.median(velocities_km_s, axis=0)
    normal /= np.linalg.norm(normal)

    time_offsets_s = np.full(len(positions_km), np.nan)
    intersections_km = np.full_like(positions_km, np.nan)

    for variant_index, (position, velocity) in enumerate(zip(positions_km, velocities_km_s)):
        if variant_index == 0 or variant_index + 1 == len(positions_km) or (variant_index + 1) % 50 == 0:
            print(
                f"  Intersecting variant {variant_index + 1}/{len(positions_km)}",
                flush=True,
            )
        try:
            _, intersection = tube_util.calcPlaneLineIntersection(
                normal,
                center_km,
                position,
                velocity,
            )
        except AssertionError:
            continue

        intersections_km[variant_index] = intersection
        delta = intersection - position
        time_offsets_s[variant_index] = np.dot(delta, velocity) / np.dot(velocity, velocity)

    return intersections_km, time_offsets_s, normal


def project_to_plane_2d(
    tube_util,
    intersections_km: np.ndarray,
    center_km: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(intersections_km).all(axis=1)
    if not np.any(valid):
        return np.empty((0, 2))

    points_for_util = intersections_km[valid].T
    return tube_util.transformPointsToXYPlane(
        points_for_util,
        center_km,
        normal,
        False,
    ).T


def plot_cutplane(
    plt,
    cutplane: CutPlane,
    points_2d_km: np.ndarray,
    output_dir: Path,
    bins: int,
    save_plot: bool,
) -> Path | None:
    print(f"  Plotting {len(points_2d_km)} projected points", flush=True)

    figure, axes = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)

    hist = axes.hist2d(
        points_2d_km[:, 0],
        points_2d_km[:, 1],
        bins=bins,
        cmap="magma",
    )
    figure.colorbar(hist[3], ax=axes, label="Variant count per bin")

    axes.scatter(
        points_2d_km[:, 0],
        points_2d_km[:, 1],
        s=6,
        c="white",
        alpha=0.35,
        linewidths=0,
    )

    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("Cut-plane x (km)")
    axes.set_ylabel("Cut-plane y (km)")
    axes.set_title(f"Cut plane {cutplane.index:04d}: {cutplane.time_utc}")

    if save_plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"cutplane_{cutplane.index:04d}_{safe_time(cutplane.time_utc)}.png"
        print(f"  Saving plot: {output_path}", flush=True)
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        return output_path

    print("  Showing plot window", flush=True)
    plt.show()
    return None


def safe_time(time_utc: str) -> str:
    return time_utc.replace(":", "").replace(".", "_")


def parse_indices(raw_indices: str | None, count: int) -> list[int]:
    if raw_indices:
        indices = [int(value.strip()) for value in raw_indices.split(",") if value.strip()]
    else:
        indices = list(range(min(5, count)))

    return sorted(set(index for index in indices if 0 <= index < count))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tube-json", type=Path, default=DEFAULT_TUBE_JSON)
    parser.add_argument("--variant-kernels", type=Path, default=DEFAULT_VARIANT_KERNELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma-separated cut-plane indices to plot. Default: first five cut planes.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Use only the first N variant kernels. Useful for quick smoke tests.",
    )
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins per axis.")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to --output-dir instead of showing them interactively.",
    )
    args = parser.parse_args()

    plt, spice, tube_util = import_runtime_dependencies()

    cutplanes = load_cutplanes(args.tube_json)
    selected_indices = parse_indices(args.indices, len(cutplanes))
    variant_kernels = variant_kernel_paths(args.variant_kernels, args.max_variants)

    print(f"Loaded {len(cutplanes)} cut planes from {args.tube_json}", flush=True)
    print(f"Using {len(variant_kernels)} variant kernels from {args.variant_kernels}", flush=True)
    print(f"Selected cut-plane indices: {selected_indices}", flush=True)

    try:
        print("Loading SPICE kernels...", flush=True)
        furnish_kernels(spice, variant_kernels)
        print("Finished loading SPICE kernels.", flush=True)

        for plot_number, index in enumerate(selected_indices, start=1):
            cutplane = cutplanes[index]
            print(
                f"\nCut plane {plot_number}/{len(selected_indices)} "
                f"(index {cutplane.index}, time {cutplane.time_utc})",
                flush=True,
            )
            et = spice.str2et(cutplane.time_utc)
            print("  Reading variant states from SPICE...", flush=True)
            positions_km, velocities_km_s, target_ids = states_at_cutplane(
                spice,
                tube_util,
                et,
                variant_kernels,
            )
            print(f"  Loaded {len(target_ids)} variant states.", flush=True)

            print("  Computing plane intersections...", flush=True)
            intersections_km, time_offsets_s, normal = intersect_variants_with_plane(
                tube_util,
                cutplane.center_km,
                positions_km,
                velocities_km_s,
            )
            print("  Projecting intersections to cut-plane coordinates...", flush=True)
            points_2d_km = project_to_plane_2d(
                tube_util,
                intersections_km,
                cutplane.center_km,
                normal,
            )
            if len(points_2d_km) == 0:
                print(f"Skipping cut plane {index}: no valid intersection points.")
                continue

            output_path = plot_cutplane(
                plt,
                cutplane,
                points_2d_km,
                args.output_dir,
                args.bins,
                args.save_plots,
            )

            finite_offsets = time_offsets_s[np.isfinite(time_offsets_s)]
            offset_summary = "no finite time offsets"
            if len(finite_offsets) > 0:
                offset_summary = (
                    f"time offset range: {finite_offsets.min():.3f}s "
                    f"to {finite_offsets.max():.3f}s"
                )
            print(
                f"Finished cut plane {cutplane.index}; used {len(points_2d_km)} points "
                f"from {len(target_ids)} variants. {offset_summary}"
            )
            if output_path is not None:
                print(f"  Wrote {output_path}", flush=True)
    finally:
        spice.kclear()


if __name__ == "__main__":
    main()

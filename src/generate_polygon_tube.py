"""Regenerate an OpenSpace polygon tube and its matching textures.

Run from the repository root, for example:

    python src/generate_polygon_tube.py
    python src/generate_polygon_tube.py --all-polygons \
        --output "data/B612_data/Test/2004 MN4/tube_2004_MN4_polygon_all.json" \
        --texture-directory "data/B612_data/Test/2004 MN4/textures_polygon_all"

The default command writes ``tube_2004_MN4_polygon.json`` and regenerates all
of its ``.osimg`` files in ``textures_polygon``. Existing files with matching
names are replaced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tube_generation.polygon import (
    POLYGON_COVERAGE_RELATIVE_TOLERANCE,
    createPolygonsFromStateCache,
)
from tube_generation import texture
from tube_generation.tube import writePolygonTube


DEFAULT_TUBE = (
    REPO_ROOT / "data" / "B612_data" / "Test" / "2004 MN4" / "tube_2004_MN4.json"
)
DEFAULT_CACHE = REPO_ROOT / "revision" / "cache" / "2004_MN4_cutplane_states.npz"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "B612_data"
    / "Test"
    / "2004 MN4"
    / "tube_2004_MN4_polygon.json"
)
DEFAULT_TEXTURE_DIRECTORY = (
    REPO_ROOT
    / "data"
    / "B612_data"
    / "Test"
    / "2004 MN4"
    / "textures_polygon"
)


def write_polygon_tube_metadata(
    output_path: str | Path,
    *,
    source_tube_path: Path,
    state_cache_path: Path,
    texture_directory: Path,
    texture_resolution: int,
    requested_alpha: float,
    selected_alpha: float,
    alpha_step: float,
    fixed_alpha: bool,
    normality_alpha: float,
    minimum_required_coverage: float,
    minimum_observed_coverage: float,
    start_polygon_index: int,
    end_polygon_index: int,
    polygon_ring_count: int,
    texture_count: int,
    missing_texture_variant_indices: dict[int, np.ndarray],
    bfo_center_inside_polygon: dict[int, bool],
    polygon_scope: str,
) -> Path:
    """Write polygon-generation settings and results beside the tube JSON."""

    output_path = Path(output_path)
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.txt")
    metadata = {
        "output_tube": str(output_path.resolve()),
        "source_tube": str(source_tube_path.resolve()),
        "state_cache": str(state_cache_path.resolve()),
        "texture_directory": str(texture_directory.resolve()),
        "texture_resolution": texture_resolution,
        "alpha_selection": "fixed" if fixed_alpha else "global auto-lower",
        "requested_maximum_alpha": requested_alpha,
        "selected_global_alpha": selected_alpha,
        "alpha_step": alpha_step,
        "normality_test_alpha": normality_alpha,
        "minimum_required_coverage": minimum_required_coverage,
        "minimum_observed_coverage": minimum_observed_coverage,
        "texture_coverage_relative_tolerance": (
            POLYGON_COVERAGE_RELATIVE_TOLERANCE
        ),
        "start_polygon_index": start_polygon_index,
        "end_polygon_index": end_polygon_index,
        "polygon_ring_count": polygon_ring_count,
        "texture_count": texture_count,
        "polygon_scope": polygon_scope,
        "position_channel_background": 0,
        "position_channel_variant": 1,
        "position_channel_fitted_mvee_center": "unused",
        "position_channel_bfo_center": 3,
        "position_channel_variant_marker_size_pixels": "5x5",
        "position_channel_bfo_marker_size_pixels": "9x9",
        "bfo_marker_rule": "paint only when BFO is covered by sampled polygon",
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        for key, value in metadata.items():
            file.write(f"{key}: {value}\n")
        file.write("\nmissing_texture_variants_by_source_polygon:\n")
        for polygon_index in range(start_polygon_index, end_polygon_index + 1):
            missing = np.asarray(
                missing_texture_variant_indices[polygon_index], dtype=int
            )
            indices = ", ".join(str(index) for index in missing)
            file.write(
                f"polygon {polygon_index}: count={len(missing)}; indices=[{indices}]\n"
            )
        omitted_bfo_markers = [
            polygon_index
            for polygon_index in range(start_polygon_index, end_polygon_index + 1)
            if not bfo_center_inside_polygon[polygon_index]
        ]
        file.write("\nomitted_bfo_texture_markers:\n")
        file.write(f"count: {len(omitted_bfo_markers)}\n")
        file.write(
            "source_polygon_indices: ["
            + ", ".join(str(index) for index in omitted_bfo_markers)
            + "]\n"
        )
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate a polygon tube JSON and its corrected UV textures."
    )
    parser.add_argument(
        "--source-tube",
        type=Path,
        default=DEFAULT_TUBE,
        help=f"Source ellipse-tube JSON. Default: {DEFAULT_TUBE}",
    )
    parser.add_argument(
        "--state-cache",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Cached variant states. Default: {DEFAULT_CACHE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Regenerated polygon-tube JSON. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--texture-directory",
        type=Path,
        default=DEFAULT_TEXTURE_DIRECTORY,
        help="Directory for regenerated polygon .osimg textures.",
    )
    parser.add_argument(
        "--texture-resolution",
        type=int,
        default=500,
        help="Square polygon texture resolution. Default: 500.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Maximum standardized alpha. Default: 1.0.",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.1,
        help="Step used while lowering alpha globally. Default: 0.1.",
    )
    parser.add_argument(
        "--fixed-alpha",
        action="store_true",
        help="Use --alpha exactly instead of automatically lowering it.",
    )
    parser.add_argument("--normality-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-coverage", type=float, default=0.99)
    polygon_start_group = parser.add_mutually_exclusive_group()
    polygon_start_group.add_argument(
        "--start-polygon-index",
        type=int,
        default=None,
        help="Override automatic first-non-Gaussian detection.",
    )
    polygon_start_group.add_argument(
        "--all-polygons",
        action="store_true",
        help=(
            "Generate from source polygon 0 instead of starting at the first "
            "non-Gaussian polygon."
        ),
    )
    args = parser.parse_args()
    if args.texture_resolution <= 0:
        raise ValueError("texture resolution must be positive")
    if not args.source_tube.is_file():
        raise FileNotFoundError(f"source tube does not exist: {args.source_tube}")
    if not args.state_cache.is_file():
        raise FileNotFoundError(f"state cache does not exist: {args.state_cache}")
    if args.output.resolve() == args.source_tube.resolve():
        raise ValueError(
            "--output must differ from --source-tube so the source ellipse tube "
            "is not overwritten"
        )

    with args.source_tube.open("r", encoding="utf-8") as file:
        source_tube = json.load(file)
    with np.load(args.state_cache, allow_pickle=False) as cache:
        required = {
            "cutplane_indices",
            "times_utc",
            "positions_km",
            "velocities_km_s",
        }
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(
                "state cache is missing required arrays: " + ", ".join(sorted(missing))
            )
        cutplane_indices = cache["cutplane_indices"].copy()
        times_utc = cache["times_utc"].copy()
        positions_km = cache["positions_km"].copy()
        velocities_km_s = cache["velocities_km_s"].copy()

    print(f"Source tube: {args.source_tube}")
    print(f"State cache: {args.state_cache}")
    print(f"Maximum standardized alpha: {args.alpha}")
    requested_start_polygon_index = (
        0 if args.all_polygons else args.start_polygon_index
    )
    polygon_scope = (
        "all usable polygons"
        if args.all_polygons
        else (
            f"explicit start at polygon {args.start_polygon_index}"
            if args.start_polygon_index is not None
            else "first non-Gaussian polygon onward"
        )
    )
    rings = createPolygonsFromStateCache(
        source_tube,
        cutplane_indices,
        times_utc,
        positions_km,
        velocities_km_s,
        alpha=args.alpha,
        normality_alpha=args.normality_alpha,
        minimum_coverage=args.minimum_coverage,
        start_polygon_index=requested_start_polygon_index,
        auto_lower_alpha=not args.fixed_alpha,
        alpha_step=args.alpha_step,
    )

    # Both the texture files and tube JSON are regenerated on every run. The
    # polygon-specific UV calculation is performed by
    # createPolygonsFromStateCache and serialized into the JSON below.
    args.texture_directory.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Regenerating polygon textures: {args.texture_directory}")
    generated_texture_count = 0
    for source_index in range(
        rings.start_polygon_index,
        rings.end_polygon_index + 1,
    ):
        source_texture_name = source_tube["polygons"][source_index]["texture"]
        texture_path = Path(source_texture_name)
        if texture_path.suffix.lower() != ".osimg" or not texture_path.stem.isdigit():
            raise ValueError(
                f"polygon {source_index} has unsupported texture filename "
                f"{source_texture_name!r}; expected a numeric .osimg filename"
            )

        bfo_is_inside = rings.bfo_center_inside_polygon[source_index]
        generated_name = texture.generateTexture(
            rings.texture_ranges_x[source_index],
            rings.texture_ranges_y[source_index],
            args.texture_resolution,
            args.texture_directory,
            int(texture_path.stem),
            rings.texture_coordinates[source_index].T,
            rings.variants_2d_km[source_index].T,
            rings.time_offsets_s[source_index],
            position_marker_points_2D=(
                np.zeros((2, 1), dtype=float) if bfo_is_inside else None
            ),
            position_marker_values=(np.array([3.0]) if bfo_is_inside else None),
            position_marker_size=9,
        )
        if generated_name != source_texture_name:
            raise RuntimeError(
                f"generated texture {generated_name!r}, expected "
                f"{source_texture_name!r}"
            )
        generated_texture_count += 1

    output_path = writePolygonTube(
        args.output.name,
        args.output.parent,
        source_tube,
        rings.rings_3d_m,
        rings.start_polygon_index,
        rings.end_polygon_index,
        rings.texture_coordinates,
        rings.center_texture_coordinates,
    )
    minimum_observed_coverage = min(item.coverage for item in rings.diagnostics)
    metadata_path = write_polygon_tube_metadata(
        output_path,
        source_tube_path=args.source_tube,
        state_cache_path=args.state_cache,
        texture_directory=args.texture_directory,
        texture_resolution=args.texture_resolution,
        requested_alpha=args.alpha,
        selected_alpha=rings.alpha,
        alpha_step=args.alpha_step,
        fixed_alpha=args.fixed_alpha,
        normality_alpha=args.normality_alpha,
        minimum_required_coverage=args.minimum_coverage,
        minimum_observed_coverage=minimum_observed_coverage,
        start_polygon_index=rings.start_polygon_index,
        end_polygon_index=rings.end_polygon_index,
        polygon_ring_count=len(rings.rings_3d_m),
        texture_count=generated_texture_count,
        missing_texture_variant_indices=rings.missing_texture_variant_indices,
        bfo_center_inside_polygon=rings.bfo_center_inside_polygon,
        polygon_scope=polygon_scope,
    )
    print(f"First polygon index: {rings.start_polygon_index}")
    print(f"Final polygon index: {rings.end_polygon_index}")
    print(f"Polygon scope: {polygon_scope}")
    print(f"Selected global alpha: {rings.alpha}")
    print(f"Polygon rings written: {len(rings.rings_3d_m)}")
    print(f"Polygon textures written: {generated_texture_count}")
    print(f"Minimum observed coverage: {minimum_observed_coverage:.6f}")
    print(f"Polygon textures: {args.texture_directory}")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

"""Generate an OpenSpace polygon-only uncertainty tube from a state cache.

Run from the repository root, for example:

    python src/generate_polygon_tube.py
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

from tube_generation.polygon import createPolygonsFromStateCache
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tube", type=Path, default=DEFAULT_TUBE)
    parser.add_argument("--state-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    parser.add_argument(
        "--start-polygon-index",
        type=int,
        default=None,
        help="Override automatic first-non-Gaussian detection.",
    )
    args = parser.parse_args()
    if args.texture_resolution <= 0:
        raise ValueError("texture resolution must be positive")

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
    rings = createPolygonsFromStateCache(
        source_tube,
        cutplane_indices,
        times_utc,
        positions_km,
        velocities_km_s,
        alpha=args.alpha,
        normality_alpha=args.normality_alpha,
        minimum_coverage=args.minimum_coverage,
        start_polygon_index=args.start_polygon_index,
        auto_lower_alpha=not args.fixed_alpha,
        alpha_step=args.alpha_step,
    )

    args.texture_directory.mkdir(parents=True, exist_ok=True)
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

        generated_name = texture.generateTexture(
            rings.texture_ranges_x[source_index],
            rings.texture_ranges_y[source_index],
            args.texture_resolution,
            args.texture_directory,
            int(texture_path.stem),
            rings.texture_coordinates[source_index].T,
            rings.variants_2d_km[source_index].T,
            rings.time_offsets_s[source_index],
        )
        if generated_name != source_texture_name:
            raise RuntimeError(
                f"generated texture {generated_name!r}, expected "
                f"{source_texture_name!r}"
            )

    output_path = writePolygonTube(
        args.output.name,
        args.output.parent,
        source_tube,
        rings.rings_3d_m,
        rings.start_polygon_index,
        rings.end_polygon_index,
        rings.texture_coordinates,
    )
    minimum_observed_coverage = min(item.coverage for item in rings.diagnostics)
    print(f"First polygon index: {rings.start_polygon_index}")
    print(f"Final polygon index: {rings.end_polygon_index}")
    print(f"Selected global alpha: {rings.alpha}")
    print(f"Polygon rings written: {len(rings.rings_3d_m)}")
    print(f"Minimum observed coverage: {minimum_observed_coverage:.6f}")
    print(f"Polygon textures: {args.texture_directory}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

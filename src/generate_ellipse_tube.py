"""Regenerate an ellipse uncertainty tube from cached propagated states.

This bypasses orbit propagation.  It reads variant positions and velocities at
the source tube's cut-plane times, intersects their local trajectories with
each cut plane, fits a two-dimensional minimum-volume enclosing ellipse, and
regenerates the tube geometry and OpenSpace textures.

Run from the repository root, for example:

    python src/generate_ellipse_tube.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull, QhullError


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from density_tube_generation.pipeline import (
    intersect_variants_with_plane,
    project_to_plane_2d,
)
from tube_generation import texture
from tube_generation import util as tube_util
from tube_generation.tube import writePolygonTube


DEFAULT_SOURCE_TUBE = (
    REPO_ROOT / "data" / "B612_data" / "Test" / "2004 MN4" / "tube_2004_MN4.json"
)
DEFAULT_STATE_CACHE = (
    REPO_ROOT / "revision" / "cache" / "2004_MN4_cutplane_states.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "B612_data"
    / "Test"
    / "2004 MN4"
    / "tube_2004_MN4_ellipse.json"
)
DEFAULT_TEXTURE_DIRECTORY = (
    REPO_ROOT
    / "data"
    / "B612_data"
    / "Test"
    / "2004 MN4"
    / "textures_ellipse"
)


@dataclass(frozen=True)
class EnclosingEllipse:
    center_2d_km: np.ndarray
    radii_km: np.ndarray
    rotation: np.ndarray
    maximum_quadratic_form: float
    fit_iterations: int


def _minimum_enclosing_ellipse(
    points_2d_km: npt.ArrayLike,
    *,
    tolerance: float = 1e-3,
    max_iterations: int = 20_000,
) -> EnclosingEllipse:
    """Fit a validated 2D MVEE with the Khachiyan algorithm.

    The legacy ellipse path interprets the bundled MVEE factor incorrectly:
    for example, unit-circle samples produce a radius of ``sqrt(2)``.  This
    implementation constructs the conventional quadratic form directly,
    ``(x-c).T @ A @ (x-c) <= 1``, and validates every original point.
    """

    points = np.asarray(points_2d_km, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("points_2d_km must have shape (N, 2), N >= 3")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_2d_km must contain only finite values")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    try:
        fit_points = points[ConvexHull(points).vertices]
    except QhullError as error:
        raise ValueError("cut-plane points do not span a two-dimensional ellipse") from error

    offset = np.mean(fit_points, axis=0)
    scale = np.std(fit_points, axis=0)
    scale_floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(fit_points))))
    if np.any(scale <= scale_floor):
        raise ValueError("cut-plane points are numerically collinear")
    standardized = (fit_points - offset) / scale

    dimension = 2
    count = len(standardized)
    lifted = np.vstack([standardized.T, np.ones(count)])
    weights = np.full(count, 1.0 / count)

    for iteration in range(1, max_iterations + 1):
        moment = (lifted * weights) @ lifted.T
        try:
            inverse_moment = np.linalg.inv(moment)
        except np.linalg.LinAlgError as error:
            raise ValueError("MVEE moment matrix is singular") from error
        leverage = np.einsum("ij,ji->i", lifted.T @ inverse_moment, lifted)
        maximum_index = int(np.argmax(leverage))
        maximum = float(leverage[maximum_index])
        if maximum <= (dimension + 1.0) * (1.0 + tolerance):
            break
        step = (maximum - dimension - 1.0) / (
            (dimension + 1.0) * (maximum - 1.0)
        )
        step = float(np.clip(step, 0.0, 1.0))
        updated = (1.0 - step) * weights
        updated[maximum_index] += step
        if np.linalg.norm(updated - weights) <= tolerance:
            weights = updated
            break
        weights = updated
    else:
        raise RuntimeError(
            f"MVEE did not converge within {max_iterations} iterations"
        )

    center_standardized = standardized.T @ weights
    covariance = (
        (standardized.T * weights) @ standardized
        - np.outer(center_standardized, center_standardized)
    )
    try:
        quadratic_standardized = np.linalg.inv(covariance) / dimension
    except np.linalg.LinAlgError as error:
        raise ValueError("MVEE covariance matrix is singular") from error

    inverse_scale = np.diag(1.0 / scale)
    quadratic = inverse_scale @ quadratic_standardized @ inverse_scale
    center = offset + scale * center_standardized

    centered = points - center
    quadratic_values = np.einsum(
        "ni,ij,nj->n", centered, quadratic, centered
    )
    maximum_quadratic_form = float(np.max(quadratic_values))
    if not np.isfinite(maximum_quadratic_form) or maximum_quadratic_form <= 0.0:
        raise ValueError("fitted ellipse has an invalid quadratic form")

    # Numerical convergence can leave boundary points microscopically outside.
    # Expand only enough to guarantee containment, plus a tiny roundoff margin.
    containment_scale = max(1.0, np.sqrt(maximum_quadratic_form)) * (1.0 + 1e-10)
    eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("fitted ellipse is not positive definite")
    radii = containment_scale / np.sqrt(eigenvalues)
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    rotation = eigenvectors[:, order]

    return EnclosingEllipse(
        center_2d_km=center,
        radii_km=radii,
        rotation=rotation,
        maximum_quadratic_form=maximum_quadratic_form,
        fit_iterations=iteration,
    )


def _solar_start_direction_2d(
    plane_normal_3d: npt.ArrayLike,
    solar_system_normal_3d: npt.ArrayLike = (0.0, 0.0, 1.0),
) -> np.ndarray:
    normal = np.asarray(plane_normal_3d, dtype=float)
    normal /= np.linalg.norm(normal)
    solar = np.asarray(solar_system_normal_3d, dtype=float)
    solar /= np.linalg.norm(solar)
    solar_on_plane = solar - np.dot(solar, normal) * normal
    length = np.linalg.norm(solar_on_plane)
    if length <= np.finfo(float).eps * 32.0:
        return np.array([1.0, 0.0])
    solar_on_plane /= length
    rotation = tube_util.calcRotationMatrix(
        normal, np.array([0.0, 0.0, 1.0])
    )
    direction = (rotation @ solar_on_plane)[:2]
    return direction / np.linalg.norm(direction)


def _sample_ellipse_equal_arc(
    ellipse: EnclosingEllipse,
    num_samples: int,
    plane_normal_3d: npt.ArrayLike,
) -> np.ndarray:
    if num_samples < 3:
        raise ValueError("num_samples must be at least three")

    dense_count = max(8192, num_samples * 128)
    theta = np.linspace(0.0, 2.0 * np.pi, dense_count + 1)
    standard = np.column_stack(
        [ellipse.radii_km[0] * np.cos(theta), ellipse.radii_km[1] * np.sin(theta)]
    )
    dense = standard @ ellipse.rotation.T + ellipse.center_2d_km
    segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    targets = np.linspace(0.0, cumulative[-1], num_samples, endpoint=False)
    sampled = np.column_stack(
        [np.interp(targets, cumulative, dense[:, axis]) for axis in range(2)]
    )

    # Keep the original tube convention: clockwise, starting toward the
    # projected Solar System normal.
    signed_area = 0.5 * np.sum(
        sampled[:, 0] * np.roll(sampled[:, 1], -1)
        - sampled[:, 1] * np.roll(sampled[:, 0], -1)
    )
    if signed_area > 0.0:
        sampled = sampled[::-1]
    radial = sampled - ellipse.center_2d_km
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    start_index = int(np.argmax(radial @ _solar_start_direction_2d(plane_normal_3d)))
    return np.roll(sampled, -start_index, axis=0)


def _ellipse_bounds(ellipse: EnclosingEllipse) -> tuple[np.ndarray, np.ndarray]:
    half_extents = np.sqrt(
        np.sum((ellipse.rotation * ellipse.radii_km[None, :]) ** 2, axis=1)
    )
    return (
        np.array(
            [ellipse.center_2d_km[0] - half_extents[0],
             ellipse.center_2d_km[0] + half_extents[0]]
        ),
        np.array(
            [ellipse.center_2d_km[1] - half_extents[1],
             ellipse.center_2d_km[1] + half_extents[1]]
        ),
    )


def _boundary_densities(
    boundary_2d_km: np.ndarray,
    variants_2d_km: np.ndarray,
) -> np.ndarray:
    """Reproduce the legacy normalized inverse-distance boundary density."""

    accumulated = np.linalg.norm(
        boundary_2d_km[:, None, :] - variants_2d_km[None, :, :], axis=2
    ).sum(axis=1)
    density_range = float(np.ptp(accumulated))
    if density_range <= np.finfo(float).eps * max(1.0, float(np.max(accumulated))):
        return np.ones(len(boundary_2d_km), dtype=float)
    return 1.0 - (accumulated - np.min(accumulated)) / density_range


def _write_metadata(
    output_path: Path,
    *,
    source_tube: Path,
    state_cache: Path,
    texture_directory: Path,
    texture_resolution: int,
    slice_count: int,
    maximum_containment_value: float,
    tolerance: float,
    skipped_trailing_cache_slices: int,
) -> Path:
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.txt")
    values = {
        "output_tube": output_path.resolve(),
        "source_tube": source_tube.resolve(),
        "state_cache": state_cache.resolve(),
        "texture_directory": texture_directory.resolve(),
        "texture_resolution": texture_resolution,
        "slice_count": slice_count,
        "fit": "2D Khachiyan minimum-volume enclosing ellipse",
        "fit_tolerance": tolerance,
        "maximum_pre_expansion_quadratic_form": maximum_containment_value,
        "position_channel_background": 0,
        "position_channel_variant": 1,
        "position_channel_fitted_mvee_center": 2,
        "position_channel_bfo_center": 3,
        "position_channel_overlap_precedence": "BFO center",
        "position_channel_variant_marker_size_pixels": "5x5",
        "position_channel_center_marker_size_pixels": "9x9",
        "orbit_propagation_performed": False,
        "skipped_trailing_cache_slices": skipped_trailing_cache_slices,
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        for key, value in values.items():
            file.write(f"{key}: {value}\n")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tube", type=Path, default=DEFAULT_SOURCE_TUBE)
    parser.add_argument("--state-cache", type=Path, default=DEFAULT_STATE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--texture-directory", type=Path, default=DEFAULT_TEXTURE_DIRECTORY
    )
    parser.add_argument("--texture-resolution", type=int, default=500)
    parser.add_argument("--fit-tolerance", type=float, default=1e-3)
    parser.add_argument("--fit-max-iterations", type=int, default=20_000)
    args = parser.parse_args()

    if args.texture_resolution <= 0:
        raise ValueError("texture resolution must be positive")
    if not args.source_tube.is_file():
        raise FileNotFoundError(f"source tube does not exist: {args.source_tube}")
    if not args.state_cache.is_file():
        raise FileNotFoundError(f"state cache does not exist: {args.state_cache}")
    if args.output.resolve() == args.source_tube.resolve():
        raise ValueError("--output must not overwrite --source-tube")

    with args.source_tube.open("r", encoding="utf-8") as file:
        source_tube = json.load(file)
    with np.load(args.state_cache, allow_pickle=False) as cache:
        required = {
            "cutplane_indices", "times_utc", "positions_km", "velocities_km_s"
        }
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(
                "state cache is missing required arrays: " + ", ".join(sorted(missing))
            )
        cutplane_indices = cache["cutplane_indices"].copy()
        times_utc = cache["times_utc"].astype(str)
        positions_km = cache["positions_km"].copy()
        velocities_km_s = cache["velocities_km_s"].copy()

    polygons = source_tube.get("polygons")
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("source tube must contain a non-empty polygons list")
    if cutplane_indices.ndim != 1 or len(cutplane_indices) != len(times_utc):
        raise ValueError("cache indices and times must be matching one-dimensional arrays")
    if positions_km.shape != velocities_km_s.shape:
        raise ValueError("cached positions and velocities must have matching shapes")
    if positions_km.ndim != 3 or positions_km.shape[0] != len(cutplane_indices):
        raise ValueError("cached state arrays have incompatible dimensions")
    if len(np.unique(cutplane_indices)) != len(cutplane_indices):
        raise ValueError("cache contains duplicate source polygon indices")

    finite_state_counts = np.count_nonzero(
        np.all(np.isfinite(positions_km), axis=2)
        & np.all(np.isfinite(velocities_km_s), axis=2),
        axis=1,
    )
    usable_cache_rows = np.flatnonzero(finite_state_counts >= 3)
    if len(usable_cache_rows) == 0:
        raise ValueError("state cache contains no slices with at least three states")
    final_usable_cache_row = int(usable_cache_rows[-1])
    invalid_interior_rows = np.flatnonzero(
        finite_state_counts[: final_usable_cache_row + 1] < 3
    )
    if len(invalid_interior_rows):
        first = int(invalid_interior_rows[0])
        raise ValueError(
            f"cache row {first} / polygon {int(cutplane_indices[first])} has fewer "
            "than three valid states inside the usable interval"
        )
    selected_cache_rows = range(final_usable_cache_row + 1)
    skipped_trailing_cache_slices = len(cutplane_indices) - len(selected_cache_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.texture_directory.mkdir(parents=True, exist_ok=True)

    rings: dict[int, np.ndarray] = {}
    boundary_uvs: dict[int, np.ndarray] = {}
    center_uvs: dict[int, np.ndarray] = {}
    fitted_centers_3d_m: dict[int, np.ndarray] = {}
    bfo_centers_3d_m: dict[int, np.ndarray] = {}
    bfo_center_uvs: dict[int, np.ndarray] = {}
    boundary_densities: dict[int, np.ndarray] = {}
    containment_values = []

    print(f"Source tube: {args.source_tube}")
    print(f"State cache: {args.state_cache}")
    print(f"Regenerating {len(selected_cache_rows)} ellipse slices")
    if skipped_trailing_cache_slices:
        print(
            f"Skipping {skipped_trailing_cache_slices} trailing cache slice(s) "
            "without sufficient finite states"
        )

    for cache_index in selected_cache_rows:
        raw_polygon_index = cutplane_indices[cache_index]
        polygon_index = int(raw_polygon_index)
        if not 0 <= polygon_index < len(polygons):
            raise ValueError(f"cache polygon index {polygon_index} is outside source tube")
        polygon = polygons[polygon_index]
        if str(polygon["time"]) != str(times_utc[cache_index]):
            raise ValueError(
                f"cache time {times_utc[cache_index]!r} does not match source polygon "
                f"{polygon_index} time {polygon['time']!r}"
            )
        center_data = polygon["center"]
        center_km = np.array(
            [center_data["x"], center_data["y"], center_data["z"]], dtype=float
        ) / 1000.0
        valid = (
            np.all(np.isfinite(positions_km[cache_index]), axis=1)
            & np.all(np.isfinite(velocities_km_s[cache_index]), axis=1)
        )
        if np.count_nonzero(valid) < 3:
            raise ValueError(f"polygon {polygon_index} has fewer than three valid states")
        intersections, time_offsets_s, normal = intersect_variants_with_plane(
            center_km,
            positions_km[cache_index, valid],
            velocities_km_s[cache_index, valid],
        )
        points_2d_km, intersection_valid = project_to_plane_2d(
            intersections, center_km, normal
        )
        time_offsets_s = time_offsets_s[intersection_valid]
        if len(points_2d_km) < 3:
            raise ValueError(f"polygon {polygon_index} has fewer than three intersections")

        fitted = _minimum_enclosing_ellipse(
            points_2d_km,
            tolerance=args.fit_tolerance,
            max_iterations=args.fit_max_iterations,
        )
        samples_2d_km = _sample_ellipse_equal_arc(
            fitted, len(polygon["points"]), normal
        )
        range_x, range_y = _ellipse_bounds(fitted)
        uv, _, _ = texture.calculateTextureCoordinates(
            samples_2d_km, range_x=range_x, range_y=range_y
        )
        # The mesh center is the fitted ellipse center, not the BFO cut-plane
        # origin. Its texture coordinate is exactly the center of the ellipse's
        # symmetric texture bounds. Preserve the BFO as separate XYZ/UV metadata.
        center_uvs[polygon_index] = np.array([0.5, 0.5])
        bfo_uv = np.array(
            [
                1.0 - (0.0 - range_x[0]) / (range_x[1] - range_x[0]),
                (0.0 - range_y[0]) / (range_y[1] - range_y[0]),
            ]
        )
        if np.any(bfo_uv < 0.0) or np.any(bfo_uv > 1.0):
            raise ValueError(
                f"polygon {polygon_index} BFO lies outside the fitted ellipse "
                "texture bounds"
            )
        bfo_center_uvs[polygon_index] = bfo_uv
        fitted_center_3d_km = tube_util.invTransformPointsToXYPlane(
            fitted.center_2d_km.reshape(2, 1), center_km, normal, False
        )[:, 0]
        fitted_centers_3d_m[polygon_index] = fitted_center_3d_km * 1000.0
        bfo_centers_3d_m[polygon_index] = center_km * 1000.0
        boundary_densities[polygon_index] = _boundary_densities(
            samples_2d_km, points_2d_km
        )

        ring_3d_km = tube_util.invTransformPointsToXYPlane(
            samples_2d_km.T, center_km, normal, False
        ).T
        rings[polygon_index] = ring_3d_km * 1000.0
        boundary_uvs[polygon_index] = uv

        texture_name = polygon["texture"]
        texture_path = Path(texture_name)
        if texture_path.suffix.lower() != ".osimg" or not texture_path.stem.isdigit():
            raise ValueError(
                f"polygon {polygon_index} has unsupported texture {texture_name!r}"
            )
        texture.generateTexture(
            range_x,
            range_y,
            args.texture_resolution,
            args.texture_directory,
            int(texture_path.stem),
            uv.T,
            points_2d_km.T,
            time_offsets_s,
            position_marker_points_2D=np.column_stack(
                (fitted.center_2d_km, np.zeros(2, dtype=float))
            ),
            position_marker_values=np.array([2.0, 3.0]),
            position_marker_size=9,
        )

        containment_values.append(fitted.maximum_quadratic_form)

        if (
            cache_index == 0
            or (cache_index + 1) % 25 == 0
            or cache_index + 1 == len(selected_cache_rows)
        ):
            print(
                f"  Completed {cache_index + 1}/{len(selected_cache_rows)} "
                f"(polygon {polygon_index}, fit iterations {fitted.fit_iterations})",
                flush=True,
            )

    selected_polygon_indices = cutplane_indices[: len(selected_cache_rows)]
    start_polygon_index = int(np.min(selected_polygon_indices))
    end_polygon_index = int(np.max(selected_polygon_indices))
    expected_indices = set(range(start_polygon_index, end_polygon_index + 1))
    if set(rings) != expected_indices:
        raise ValueError("cache polygon indices must form one contiguous interval")

    output_path = Path(
        writePolygonTube(
            args.output.name,
            args.output.parent,
            source_tube,
            rings,
            start_polygon_index,
            end_polygon_index,
            polygon_texture_coordinates=boundary_uvs,
            polygon_center_texture_coordinates=center_uvs,
            polygon_centers_3d_m=fitted_centers_3d_m,
            polygon_bfo_centers_3d_m=bfo_centers_3d_m,
            polygon_point_densities=boundary_densities,
            polygon_bfo_center_texture_coordinates=bfo_center_uvs,
        )
    )
    metadata_path = _write_metadata(
        output_path,
        source_tube=args.source_tube,
        state_cache=args.state_cache,
        texture_directory=args.texture_directory,
        texture_resolution=args.texture_resolution,
        slice_count=len(rings),
        maximum_containment_value=float(np.max(containment_values)),
        tolerance=args.fit_tolerance,
        skipped_trailing_cache_slices=skipped_trailing_cache_slices,
    )
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Textures: {args.texture_directory}")


if __name__ == "__main__":
    main()

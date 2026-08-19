"""Polygon boundaries for non-Gaussian uncertainty-tube cut planes.

This module only computes a two-dimensional boundary.  Resampling the boundary,
aligning it with adjacent rings, and transforming it back to three dimensions
are separate tube-construction steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import alphashape
import numpy as np
import numpy.typing as npt
import pingouin as pg
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.prepared import prep

from tube_generation import util as tube_util
from tube_generation import texture as tube_texture


@dataclass(frozen=True)
class AlphaShapeResult:
    """Result of fitting an alpha shape to one projected cut-plane cloud.

    Attributes:
        boundary_2d: Ordered exterior vertices with shape ``(N, 2)`` in the
            original input units.  The first vertex is not repeated at the end.
        alpha: Alpha supplied by the caller.
        coverage: Fraction of input points covered by the selected polygon.
        component_count: Number of polygon components returned by alphashape.
        selected_component_area: Area of the selected component in squared
            input units.
        used_convex_hull: Whether a convex hull was used because alphashape did
            not return a usable polygon.
        standardized: Whether the fit was performed in standardized coordinates.
    """

    boundary_2d: np.ndarray
    alpha: float
    coverage: float
    component_count: int
    selected_component_area: float
    used_convex_hull: bool
    standardized: bool


@dataclass(frozen=True)
class AlphaSelectionResult:
    """Globally selected alpha and its worst-case diagnostics.

    The selected value has been validated against every supplied cut-plane
    point cloud.  It can therefore be reused while producing the tube one slice
    at a time.
    """

    alpha: float
    minimum_coverage: float
    worst_slice_index: int
    slice_count: int
    candidate_alphas_tested: tuple[float, ...]


@dataclass(frozen=True)
class PolygonRingDiagnostic:
    """Validation information for one generated polygon ring."""

    cache_index: int
    polygon_index: int
    time_utc: str
    point_count: int
    coverage: float
    component_count: int
    boundary_vertex_count: int


@dataclass(frozen=True)
class PolygonRingsResult:
    """All 3D rings required to serialize one polygon-only tube."""

    rings_3d_m: dict[int, np.ndarray]
    start_polygon_index: int
    end_polygon_index: int
    alpha: float
    diagnostics: tuple[PolygonRingDiagnostic, ...]
    texture_coordinates: dict[int, np.ndarray]
    variants_2d_km: dict[int, np.ndarray]
    time_offsets_s: dict[int, np.ndarray]
    texture_ranges_x: dict[int, np.ndarray]
    texture_ranges_y: dict[int, np.ndarray]


def _points_as_rows(points_2d: npt.ArrayLike) -> np.ndarray:
    """Validate and return projected points with shape ``(N, 2)``."""

    points = np.asarray(points_2d, dtype=float)
    if points.ndim != 2:
        raise ValueError("points_2d must be a two-dimensional array")

    # Accept the legacy tube-generation convention (2, N), while making (N, 2)
    # the canonical and unambiguous public representation.
    if points.shape[1] == 2:
        pass
    elif points.shape[0] == 2:
        points = points.T
    else:
        raise ValueError("points_2d must have shape (N, 2) or (2, N)")

    if len(points) < 4:
        raise ValueError("at least four projected points are required")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_2d must contain only finite values")
    if np.linalg.matrix_rank(points - np.mean(points, axis=0)) < 2:
        raise ValueError("points_2d must span two dimensions")
    return points


def _polygon_components(geometry) -> list[Polygon]:
    """Recursively extract non-empty polygon components from a geometry."""

    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        components = []
        for child in geometry.geoms:
            components.extend(_polygon_components(child))
        return components
    return []


def _coverage(polygon: Polygon, points: np.ndarray) -> float:
    """Return the fraction of points inside or on the polygon boundary."""

    prepared = prep(polygon)
    covered = sum(prepared.covers(Point(point)) for point in points)
    return float(covered / len(points))


def calculate_alpha_shape(
    points_2d: npt.ArrayLike,
    alpha: float,
    *,
    standardize: bool = True,
    fallback_to_convex_hull: bool = True,
) -> AlphaShapeResult:
    """Fit an alpha-shape polygon to projected variant intersections.

    Args:
        points_2d: Projected cut-plane intersections.  The canonical shape is
            ``(num_variants, 2)``; legacy ``(2, num_variants)`` arrays are also
            accepted.  Coordinates may be in kilometres or metres, but the
            returned boundary uses the same units.
        alpha: Alpha-shape parameter.  Zero produces the convex hull; increasing
            positive values generally make the boundary tighter and more
            concave.  With ``standardize=True`` this parameter is dimensionless.
        standardize: Center and independently scale both axes before fitting.
            This makes one alpha easier to compare across differently sized cut
            planes.  The resulting boundary is transformed back to input units.
        fallback_to_convex_hull: Use the point-cloud convex hull if alphashape
            returns no polygonal component.  If false, raise ``ValueError``.

    Returns:
        An :class:`AlphaShapeResult`.  If alphashape returns disconnected
        components, the largest-area component is selected and the total number
        is recorded in ``component_count``.  Inspect ``coverage`` before using
        that boundary as a tube ring.

    Raises:
        ValueError: If the input is invalid, alpha is negative/non-finite, or no
            polygon can be constructed without an enabled fallback.
    """

    points = _points_as_rows(points_2d)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a finite non-negative number")

    center = np.zeros(2, dtype=float)
    scale = np.ones(2, dtype=float)
    fit_points = points
    if standardize:
        center = np.mean(points, axis=0)
        scale = np.std(points, axis=0)
        if np.any(scale <= np.finfo(float).eps):
            raise ValueError("cannot standardize a degenerate point cloud")
        fit_points = (points - center) / scale

    geometry = alphashape.alphashape(fit_points, alpha)
    components = _polygon_components(geometry)
    component_count = len(components)
    used_convex_hull = False

    if components:
        selected_fit_polygon = max(components, key=lambda polygon: polygon.area)
    elif fallback_to_convex_hull:
        # Shapely's MultiPoint is imported locally to keep the public imports
        # focused on output geometry types.
        from shapely.geometry import MultiPoint

        hull_geometry = MultiPoint(fit_points).convex_hull
        if not isinstance(hull_geometry, Polygon) or hull_geometry.is_empty:
            raise ValueError("the point cloud does not have a polygonal convex hull")
        selected_fit_polygon = hull_geometry
        used_convex_hull = True
    else:
        raise ValueError("alphashape returned no polygonal component")

    fit_boundary = np.asarray(selected_fit_polygon.exterior.coords[:-1], dtype=float)
    boundary = fit_boundary * scale + center
    selected_polygon = Polygon(boundary)
    if not selected_polygon.is_valid:
        selected_polygon = selected_polygon.buffer(0)
    if not isinstance(selected_polygon, Polygon) or selected_polygon.is_empty:
        raise ValueError("the selected alpha-shape boundary is not a valid polygon")

    # The existing NEOviz ellipse rings use clockwise winding.  Shapely normally
    # emits counter-clockwise exteriors, so enforce the tube convention here.
    x = boundary[:, 0]
    y = boundary[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    if signed_area > 0.0:
        boundary = boundary[::-1]
        selected_polygon = Polygon(boundary)

    return AlphaShapeResult(
        boundary_2d=boundary,
        alpha=float(alpha),
        coverage=_coverage(selected_polygon, points),
        component_count=component_count,
        selected_component_area=float(selected_polygon.area),
        used_convex_hull=used_convex_hull,
        standardized=standardize,
    )


def select_global_alpha(
    cutplane_points_2d: Iterable[npt.ArrayLike],
    *,
    candidate_alphas: Iterable[float] | None = None,
    minimum_coverage: float = 0.99,
    standardize: bool = True,
) -> AlphaSelectionResult:
    """Select one conservative alpha for a sequence of tube cut planes.

    Candidates are evaluated from largest to smallest.  The first candidate
    that produces exactly one polygon component and covers at least
    ``minimum_coverage`` of the points on *every* slice is returned.  Selecting
    the largest valid candidate gives the tightest boundary among the tested
    values while preserving one global alpha across time.

    The default candidates are all at or below 1.0.  Alpha zero is included as
    a final convex-hull candidate, making the default search robust for any
    non-degenerate two-dimensional point cloud.

    This function is intended for one-time calibration.  After selection, call
    :func:`calculate_alpha_shape` slice by slice with the returned alpha; there
    is no need to retain every slice during production tube generation.

    Args:
        cutplane_points_2d: Iterable of projected point clouds.  Each item must
            have shape ``(num_variants, 2)`` or legacy shape ``(2, num_variants)``.
        candidate_alphas: Values to test.  Values above 1.0 are allowed only
            when explicitly supplied.  Duplicates are removed and candidates
            are always tested from largest to smallest.
        minimum_coverage: Required fraction of points covered on every slice.
        standardize: Forwarded to :func:`calculate_alpha_shape`.  Keep this true
            when using one alpha across cut planes with different physical size.

    Returns:
        The selected alpha and worst-case coverage diagnostics.

    Raises:
        ValueError: If there are no slices, configuration is invalid, a slice
            is degenerate, or none of the requested candidates works globally.
    """

    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in the interval (0, 1]")

    point_clouds = [_points_as_rows(points) for points in cutplane_points_2d]
    if not point_clouds:
        raise ValueError("at least one cut-plane point cloud is required")

    if candidate_alphas is None:
        candidates = tuple(np.round(np.linspace(1.0, 0.1, 10), 10)) + (0.0,)
    else:
        raw_candidates = [float(alpha) for alpha in candidate_alphas]
        if not raw_candidates:
            raise ValueError("candidate_alphas must contain at least one value")
        if any(not np.isfinite(alpha) or alpha < 0.0 for alpha in raw_candidates):
            raise ValueError("candidate alphas must be finite and non-negative")
        candidates = tuple(sorted(set(raw_candidates), reverse=True))

    tested: list[float] = []
    failure_details: list[str] = []
    for alpha in candidates:
        tested.append(alpha)
        coverages = []
        failed_slice = None
        failure_reason = ""

        for slice_index, points in enumerate(point_clouds):
            try:
                result = calculate_alpha_shape(
                    points,
                    alpha,
                    standardize=standardize,
                    fallback_to_convex_hull=False,
                )
            except ValueError as error:
                failed_slice = slice_index
                failure_reason = str(error)
                break

            coverages.append(result.coverage)
            if result.component_count != 1:
                failed_slice = slice_index
                failure_reason = f"returned {result.component_count} components"
                break
            if result.coverage < minimum_coverage:
                failed_slice = slice_index
                failure_reason = (
                    f"coverage {result.coverage:.6f} is below {minimum_coverage:.6f}"
                )
                break

        if failed_slice is None:
            worst_slice_index = int(np.argmin(coverages))
            return AlphaSelectionResult(
                alpha=float(alpha),
                minimum_coverage=float(coverages[worst_slice_index]),
                worst_slice_index=worst_slice_index,
                slice_count=len(point_clouds),
                candidate_alphas_tested=tuple(tested),
            )

        failure_details.append(
            f"alpha={alpha:g} failed at slice {failed_slice}: {failure_reason}"
        )

    details = "; ".join(failure_details)
    raise ValueError(f"no candidate alpha satisfies the global constraints; {details}")


def _resample_closed_boundary(
    boundary_2d: npt.ArrayLike,
    num_samples: int,
) -> np.ndarray:
    """Resample a closed boundary uniformly by arc length."""

    boundary = np.asarray(boundary_2d, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2:
        raise ValueError("boundary_2d must have shape (N, 2)")
    if not np.all(np.isfinite(boundary)):
        raise ValueError("boundary_2d must contain only finite values")
    if not isinstance(num_samples, (int, np.integer)) or num_samples < 3:
        raise ValueError("num_samples must be an integer of at least three")
    if len(boundary) > 1 and np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]

    # Consecutive duplicate vertices create zero-length segments and ambiguous
    # interpolation intervals, so remove them before closing the ring.
    keep = np.ones(len(boundary), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(boundary, axis=0), axis=1) > 0.0
    boundary = boundary[keep]
    if len(boundary) < 3:
        raise ValueError("boundary_2d must contain at least three distinct vertices")

    closed = np.vstack([boundary, boundary[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = cumulative[-1]
    if not np.isfinite(perimeter) or perimeter <= 0.0:
        raise ValueError("boundary_2d must have a positive finite perimeter")

    distances = np.linspace(0.0, perimeter, num_samples, endpoint=False)
    return np.column_stack(
        [np.interp(distances, cumulative, closed[:, axis]) for axis in range(2)]
    )


def sample_alpha_shape_boundary(
    alpha_shape: AlphaShapeResult | npt.ArrayLike,
    num_samples: int,
    plane_normal_3d: npt.ArrayLike,
    *,
    solar_system_normal_3d: npt.ArrayLike = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Sample an alpha boundary using the NEOviz tube-ring convention.

    The returned points are equally spaced by boundary arc length, ordered
    clockwise, and cyclically reordered so vertex zero points most closely in
    the direction of the Solar System normal projected onto the cut plane.

    Args:
        alpha_shape: An :class:`AlphaShapeResult` or an ordered ``(N, 2)``
            boundary in cut-plane coordinates.
        num_samples: Exact number of output vertices.  The first vertex is not
            repeated at the end, matching the tube JSON convention.
        plane_normal_3d: Normal of the cut plane in the same 3D reference frame
            used to project points into ``boundary_2d``.
        solar_system_normal_3d: Solar System/ecliptic normal in that reference
            frame.  NEOviz currently uses ECLIPJ2000, whose normal is ``(0,0,1)``.

    Returns:
        Array with shape ``(num_samples, 2)`` in the boundary's original units.

    Raises:
        ValueError: If the boundary or vectors are invalid, or if the Solar
            System normal is parallel to the cut-plane normal and therefore has
            no defined direction within the plane.
    """

    boundary = (
        alpha_shape.boundary_2d
        if isinstance(alpha_shape, AlphaShapeResult)
        else np.asarray(alpha_shape, dtype=float)
    )
    plane_normal = np.asarray(plane_normal_3d, dtype=float)
    solar_normal = np.asarray(solar_system_normal_3d, dtype=float)
    if plane_normal.shape != (3,) or solar_normal.shape != (3,):
        raise ValueError("plane and Solar System normals must have shape (3,)")
    if not np.all(np.isfinite(plane_normal)) or not np.all(np.isfinite(solar_normal)):
        raise ValueError("plane and Solar System normals must contain finite values")
    plane_length = np.linalg.norm(plane_normal)
    solar_length = np.linalg.norm(solar_normal)
    if plane_length == 0.0 or solar_length == 0.0:
        raise ValueError("plane and Solar System normals must be non-zero")
    plane_normal = plane_normal / plane_length
    solar_normal = solar_normal / solar_length

    # A direction vector is rotated but never translated.  This deliberately
    # avoids the legacy ellipse helper's point-versus-vector ambiguity.
    solar_on_plane = solar_normal - np.dot(solar_normal, plane_normal) * plane_normal
    projected_length = np.linalg.norm(solar_on_plane)
    if projected_length <= np.finfo(float).eps * 32.0:
        raise ValueError(
            "Solar System normal is parallel to the cut-plane normal; "
            "the boundary start direction is undefined"
        )
    solar_on_plane /= projected_length
    rotation = tube_util.calcRotationMatrix(
        plane_normal,
        np.array([0.0, 0.0, 1.0]),
    )
    start_direction_2d = (rotation @ solar_on_plane)[:2]
    start_direction_2d /= np.linalg.norm(start_direction_2d)

    sampled = _resample_closed_boundary(boundary, num_samples)

    # Negative signed area denotes clockwise winding in a conventional XY frame.
    x = sampled[:, 0]
    y = sampled[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    if signed_area > 0.0:
        sampled = sampled[::-1]

    polygon = Polygon(boundary)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        raise ValueError("boundary_2d does not define one valid polygon")
    boundary_center = np.asarray(polygon.centroid.coords[0], dtype=float)

    radial_vectors = sampled - boundary_center
    radial_lengths = np.linalg.norm(radial_vectors, axis=1)
    valid_radial = radial_lengths > np.finfo(float).eps
    if not np.any(valid_radial):
        raise ValueError("sampled boundary has no direction from its centroid")
    direction_scores = np.full(len(sampled), -np.inf)
    direction_scores[valid_radial] = (
        radial_vectors[valid_radial] / radial_lengths[valid_radial, None]
    ) @ start_direction_2d
    starting_index = int(np.argmax(direction_scores))
    return np.roll(sampled, -starting_index, axis=0)


def _create_polygon_ring_with_samples(
    points_2d_km: npt.ArrayLike,
    plane_center_km: npt.ArrayLike,
    plane_normal_3d: npt.ArrayLike,
    alpha: float,
    num_samples: int,
    *,
    minimum_coverage: float = 0.99,
    solar_system_normal_3d: npt.ArrayLike = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, AlphaShapeResult]:
    """Create one sampled 3D polygon ring from projected cut-plane points.

    The returned coordinates are in metres for direct insertion into the tube
    JSON.  The input cut-plane coordinates and center are in kilometres.
    """

    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in the interval (0, 1]")
    center = np.asarray(plane_center_km, dtype=float)
    normal = np.asarray(plane_normal_3d, dtype=float)
    if center.shape != (3,) or normal.shape != (3,):
        raise ValueError("plane center and normal must have shape (3,)")

    shape_result = calculate_alpha_shape(
        points_2d_km,
        alpha,
        standardize=True,
        fallback_to_convex_hull=False,
    )
    if shape_result.component_count != 1:
        raise ValueError(
            f"alpha shape returned {shape_result.component_count} components; expected one"
        )
    if shape_result.coverage < minimum_coverage:
        raise ValueError(
            f"alpha-shape coverage {shape_result.coverage:.6f} is below "
            f"the required {minimum_coverage:.6f}"
        )

    samples_2d_km = sample_alpha_shape_boundary(
        shape_result,
        num_samples,
        normal,
        solar_system_normal_3d=solar_system_normal_3d,
    )
    ring_3d_km = tube_util.invTransformPointsToXYPlane(
        samples_2d_km.T,
        center,
        normal,
        False,
    ).T
    return samples_2d_km, ring_3d_km * 1000.0, shape_result


def createPolygonRing(
    points_2d_km: npt.ArrayLike,
    plane_center_km: npt.ArrayLike,
    plane_normal_3d: npt.ArrayLike,
    alpha: float,
    num_samples: int,
    *,
    minimum_coverage: float = 0.99,
    solar_system_normal_3d: npt.ArrayLike = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, AlphaShapeResult]:
    """Create a sampled 3D polygon ring while preserving the public API."""

    _, ring_3d_m, shape_result = _create_polygon_ring_with_samples(
        points_2d_km,
        plane_center_km,
        plane_normal_3d,
        alpha,
        num_samples,
        minimum_coverage=minimum_coverage,
        solar_system_normal_3d=solar_system_normal_3d,
    )
    return ring_3d_m, shape_result


def createPolygonsFromStateCache(
    source_tube: dict,
    cutplane_indices: npt.ArrayLike,
    times_utc: npt.ArrayLike,
    positions_km: npt.ArrayLike,
    velocities_km_s: npt.ArrayLike,
    *,
    alpha: float = 1.0,
    normality_alpha: float = 0.05,
    minimum_coverage: float = 0.99,
    start_polygon_index: int | None = None,
    solar_system_normal_3d: npt.ArrayLike = (0.0, 0.0, 1.0),
    auto_lower_alpha: bool = True,
    alpha_step: float = 0.1,
) -> PolygonRingsResult:
    """Generate all polygon rings from the first non-Gaussian slice onward.

    Cached positions and velocities are intersected with the centers from the
    existing tube.  If ``start_polygon_index`` is omitted, the first slice that
    fails Pingouin's Henze-Zirkler test at ``normality_alpha`` is selected.
    By default, ``alpha`` is treated as a maximum: the largest globally valid
    value at or below it is selected in ``alpha_step`` decrements and then used
    unchanged for every generated ring.
    Every source-tube polygon from that index through the end must have one
    usable cache row; missing or invalid rows raise an error instead of silently
    creating a temporal gap.
    """

    # Imported locally to keep the one-slice polygon geometry usable without
    # loading the density-analysis pipeline.
    from density_tube_generation.pipeline import (
        intersect_variants_with_plane,
        project_to_plane_2d,
    )

    if not isinstance(source_tube, dict) or "polygons" not in source_tube:
        raise ValueError("source_tube must contain a polygons list")
    polygons = source_tube["polygons"]
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("source_tube polygons must be a non-empty list")
    if not 0.0 < normality_alpha < 1.0:
        raise ValueError("normality_alpha must be in the interval (0, 1)")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and between 0 and 1")
    if not np.isfinite(alpha_step) or alpha_step <= 0.0:
        raise ValueError("alpha_step must be finite and positive")

    indices = np.asarray(cutplane_indices)
    times = np.asarray(times_utc).astype(str)
    positions = np.asarray(positions_km, dtype=float)
    velocities = np.asarray(velocities_km_s, dtype=float)
    if indices.ndim != 1 or times.ndim != 1:
        raise ValueError("cutplane_indices and times_utc must be one-dimensional")
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions_km must have shape (num_slices, num_variants, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("velocities_km_s must have the same shape as positions_km")
    if len(indices) != len(times) or len(indices) != positions.shape[0]:
        raise ValueError("cache metadata must match the cache slice dimension")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("each source polygon index may occur only once in the cache")
    if np.any(indices < 0) or np.any(indices >= len(polygons)):
        raise ValueError("cache contains an out-of-range source polygon index")

    cache_by_polygon = {int(index): cache_index for cache_index, index in enumerate(indices)}

    def project_cache_index(cache_index: int):
        polygon_index = int(indices[cache_index])
        polygon = polygons[polygon_index]
        center_data = polygon["center"]
        center_km = np.array(
            [center_data["x"], center_data["y"], center_data["z"]], dtype=float
        ) / 1000.0
        valid = (
            np.all(np.isfinite(positions[cache_index]), axis=1)
            & np.all(np.isfinite(velocities[cache_index]), axis=1)
        )
        if not np.any(valid):
            raise ValueError(
                f"cache row {cache_index} / polygon {polygon_index} has no finite states"
            )
        intersections, time_offsets_s, normal = intersect_variants_with_plane(
            center_km,
            positions[cache_index, valid],
            velocities[cache_index, valid],
        )
        points_2d_km, intersection_valid = project_to_plane_2d(
            intersections, center_km, normal
        )
        time_offsets_s = time_offsets_s[intersection_valid]
        if len(points_2d_km) < 4:
            raise ValueError(
                f"cache row {cache_index} / polygon {polygon_index} has fewer than "
                "four valid projected points"
            )
        return polygon_index, center_km, normal, points_2d_km, time_offsets_s

    if start_polygon_index is None:
        selected_start = None
        for cache_index in range(len(indices)):
            try:
                polygon_index, _, _, points_2d_km, _ = project_cache_index(cache_index)
            except ValueError:
                continue
            normality = pg.multivariate_normality(points_2d_km, alpha=normality_alpha)
            if not bool(normality.normal):
                selected_start = polygon_index
                break
        if selected_start is None:
            raise ValueError("no non-Gaussian cut plane was found in the state cache")
        start_polygon_index = selected_start
    else:
        start_polygon_index = int(start_polygon_index)
        if not 0 <= start_polygon_index < len(polygons):
            raise IndexError(
                f"start_polygon_index must be between 0 and {len(polygons) - 1}"
            )

    end_polygon_index = len(polygons) - 1
    while end_polygon_index >= start_polygon_index:
        trailing_cache_index = cache_by_polygon.get(end_polygon_index)
        if trailing_cache_index is None:
            end_polygon_index -= 1
            continue
        trailing_valid = (
            np.all(np.isfinite(positions[trailing_cache_index]), axis=1)
            & np.all(np.isfinite(velocities[trailing_cache_index]), axis=1)
        )
        if np.any(trailing_valid):
            break
        end_polygon_index -= 1
    if end_polygon_index < start_polygon_index:
        raise ValueError("no usable cached slices exist at or after the selected start")

    required_indices = range(start_polygon_index, end_polygon_index + 1)
    missing_cache_indices = [index for index in required_indices if index not in cache_by_polygon]
    if missing_cache_indices:
        raise ValueError(
            f"state cache is missing {len(missing_cache_indices)} required polygon slices; "
            f"first missing index: {missing_cache_indices[0]}"
        )

    projected_slices = {}
    for polygon_index in required_indices:
        cache_index = cache_by_polygon[polygon_index]
        _, center_km, normal, points_2d_km, time_offsets_s = project_cache_index(
            cache_index
        )
        projected_slices[polygon_index] = (
            cache_index,
            center_km,
            normal,
            points_2d_km,
            time_offsets_s,
        )

    selected_alpha = float(alpha)
    if auto_lower_alpha:
        candidate_count = int(np.floor(alpha / alpha_step + 1e-12))
        candidates = [alpha - step * alpha_step for step in range(candidate_count + 1)]
        if not candidates or candidates[-1] > np.finfo(float).eps:
            candidates.append(0.0)
        candidates = [max(0.0, float(candidate)) for candidate in candidates]
        selection = select_global_alpha(
            [projected_slices[index][3] for index in required_indices],
            candidate_alphas=candidates,
            minimum_coverage=minimum_coverage,
            standardize=True,
        )
        selected_alpha = selection.alpha

    rings: dict[int, np.ndarray] = {}
    texture_coordinates: dict[int, np.ndarray] = {}
    variants_2d_km: dict[int, np.ndarray] = {}
    time_offsets_by_polygon: dict[int, np.ndarray] = {}
    texture_ranges_x: dict[int, np.ndarray] = {}
    texture_ranges_y: dict[int, np.ndarray] = {}
    diagnostics = []
    for polygon_index in required_indices:
        cache_index, center_km, normal, points_2d_km, time_offsets_s = (
            projected_slices[polygon_index]
        )
        num_samples = len(polygons[polygon_index]["points"])
        samples_2d_km, ring_3d_m, shape_result = _create_polygon_ring_with_samples(
            points_2d_km,
            center_km,
            normal,
            selected_alpha,
            num_samples,
            minimum_coverage=minimum_coverage,
            solar_system_normal_3d=solar_system_normal_3d,
        )
        rings[polygon_index] = ring_3d_m
        # The actual tube cross-section is the polygon formed by connecting
        # these sampled points. Use that same polygon for UV bounds and texture
        # inclusion, rather than the denser alpha-shape boundary from which it
        # was sampled.
        uv, range_x, range_y = tube_texture.calculateTextureCoordinates(
            samples_2d_km
        )

        # The alpha shape deliberately permits a small number of variants to
        # remain outside the polygon. Do not clamp those variants onto the
        # texture edge, since that would create artificial density there.
        selected_polygon = prep(Polygon(samples_2d_km))
        covered = np.fromiter(
            (selected_polygon.covers(Point(point)) for point in points_2d_km),
            dtype=bool,
            count=len(points_2d_km),
        )
        texture_points_2d_km = points_2d_km[covered]
        texture_time_offsets_s = time_offsets_s[covered]
        if len(texture_points_2d_km) == 0:
            raise ValueError(
                f"polygon {polygon_index} contains no variants for texture generation"
            )
        texture_coordinates[polygon_index] = uv
        variants_2d_km[polygon_index] = texture_points_2d_km
        time_offsets_by_polygon[polygon_index] = texture_time_offsets_s
        texture_ranges_x[polygon_index] = range_x
        texture_ranges_y[polygon_index] = range_y
        diagnostics.append(
            PolygonRingDiagnostic(
                cache_index=cache_index,
                polygon_index=polygon_index,
                time_utc=str(times[cache_index]),
                point_count=len(points_2d_km),
                coverage=shape_result.coverage,
                component_count=shape_result.component_count,
                boundary_vertex_count=len(shape_result.boundary_2d),
            )
        )

    return PolygonRingsResult(
        rings_3d_m=rings,
        start_polygon_index=start_polygon_index,
        end_polygon_index=end_polygon_index,
        alpha=selected_alpha,
        diagnostics=tuple(diagnostics),
        texture_coordinates=texture_coordinates,
        variants_2d_km=variants_2d_km,
        time_offsets_s=time_offsets_by_polygon,
        texture_ranges_x=texture_ranges_x,
        texture_ranges_y=texture_ranges_y,
    )

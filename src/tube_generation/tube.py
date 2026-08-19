import os
import json
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import numpy.typing as npt

def createTube(data_dictionary, time_polygons, num_ellipse_samples):
    """
    """

    # Fill in the polygons list
    time = 0
    for time_polygon in time_polygons:
        # Time string
        data_dictionary["polygons"].append({"time": time_polygon.time})

        # Polygon center point in 3D
        data_dictionary["polygons"][time]["center"] = {
            "x": time_polygon.center[0],
            "y": time_polygon.center[1],
            "z": time_polygon.center[2]
        }

        # Texture filenames
        data_dictionary["polygons"][time]["texture"] = time_polygon.texture
        

        # Ellipsoid axes lengths
        data_dictionary["polygons"][time]["axes-lengths"] = {
            "a": time_polygon.ellipsoid.axes_lengths[0],
            "b": time_polygon.ellipsoid.axes_lengths[1],
            "c": time_polygon.ellipsoid.axes_lengths[2]
        }

        # Ellipsoid rotation matrix
        # We might have to transpose the matrix here depending on how OPenSpace vs python
        # treats matricis
        data_dictionary["polygons"][time]["rotation"] = {
            "x1": time_polygon.ellipsoid.rotation_matrix[0][0],
            "x2": time_polygon.ellipsoid.rotation_matrix[0][1],
            "x3": time_polygon.ellipsoid.rotation_matrix[0][2],
            "y1": time_polygon.ellipsoid.rotation_matrix[1][0],
            "y2": time_polygon.ellipsoid.rotation_matrix[1][1],
            "y3": time_polygon.ellipsoid.rotation_matrix[1][2],
            "z1": time_polygon.ellipsoid.rotation_matrix[2][0],
            "z2": time_polygon.ellipsoid.rotation_matrix[2][1],
            "z3": time_polygon.ellipsoid.rotation_matrix[2][2]
        }

        # The points of the polygon
        tube_polygon_points = []
        for p in range(num_ellipse_samples):
            tube_polygon_points.append({
                "x": time_polygon.samples[p].position_3D[0],
                "y": time_polygon.samples[p].position_3D[1],
                "z": time_polygon.samples[p].position_3D[2],
                "u": time_polygon.samples[p].texture_coordinate[0],
                "v": time_polygon.samples[p].texture_coordinate[1], 
                "data": {
                    "density": time_polygon.samples[p].density,
                    "standard-deviation-major": time_polygon.samples[p].standard_deviation[0],
                    "standard-deviation-minor": time_polygon.samples[p].standard_deviation[1],
                    "variance-major": time_polygon.samples[p].variance[0],
                    "variance-minor": time_polygon.samples[p].variance[1],
                    "is-gaussian": time_polygon.samples[p].is_gaussian,
                    "is-deviating": time_polygon.samples[p].is_deviating
                }
            })
        data_dictionary["polygons"][time]["points"] = tube_polygon_points

        # Increment time index
        time += 1

    return data_dictionary

def writeTube(filename, output_directory, time_polygons, configuration):
    """
    
    """
    
    # Meta data for the whole tube file
    version = [int(0), int(2)]
    texture_channels = ["Density", "Position", "Time difference", "Time difference range"]

    # File meta data and overall tube meta data
    data_dictionary = {
        "version": {
            "major": version[0],
            "minor": version[1]
        },
        "texture-channels": [],
        "polygons": []
    }

    # Fill in the texture channels
    for tc in range(len(texture_channels)):
        data_dictionary["texture-channels"].append(texture_channels[tc])

    # Fill the dictionary with the tube data
    data_dictionary = createTube(
        data_dictionary,
        time_polygons,
        configuration["num_polygon_samples"]
    )

    # Create the file
    filepath = os.path.join(output_directory, filename)
    print("Writing tube file", filepath)

    # Write the tube information to the JSON file
    with open(filepath, 'w') as fp:
        json.dump(data_dictionary, fp)


def createPolygonTube(
    source_tube: dict,
    polygon_rings_3d_m: Mapping[int, npt.ArrayLike],
    start_polygon_index: int,
    end_polygon_index: int | None = None,
    polygon_texture_coordinates: Mapping[int, npt.ArrayLike] | None = None,
) -> dict:
    """Create a polygon-only tube in the existing OpenSpace JSON schema.

    The returned tube starts at ``start_polygon_index`` and extends through
    ``end_polygon_index`` (or the source tube's final polygon when omitted).
    A replacement 3D ring is required for
    every polygon in that interval. Slice metadata and per-point data values
    are copied unchanged. Positions are replaced, and UV coordinates are also
    replaced when ``polygon_texture_coordinates`` is supplied.

    Args:
        source_tube: Parsed existing tube JSON with top-level ``version``,
            ``texture-channels``, and ``polygons`` fields.
        polygon_rings_3d_m: Mapping from the original source-tube polygon index
            to a ring with shape ``(num_polygon_samples, 3)``.  Coordinates must
            be finite and in metres, matching the existing tube JSON.
        start_polygon_index: Original source-tube index of the first polygon to
            include.  Earlier ellipse polygons are not copied into the result.
        end_polygon_index: Inclusive final source-tube index.  This permits
            omission of trailing slices outside the variant kernels' coverage.

    Returns:
        A new dictionary with exactly the same structure as the source tube but
        containing only the requested polygon interval.

    Raises:
        TypeError: If the source tube or ring collection has the wrong type.
        ValueError: If required schema fields, rings, or coordinates are invalid.
        IndexError: If ``start_polygon_index`` is outside the source tube.
    """

    if not isinstance(source_tube, dict):
        raise TypeError("source_tube must be a parsed JSON dictionary")
    if not isinstance(polygon_rings_3d_m, Mapping):
        raise TypeError("polygon_rings_3d_m must map source polygon indices to rings")
    if polygon_texture_coordinates is not None and not isinstance(
        polygon_texture_coordinates, Mapping
    ):
        raise TypeError(
            "polygon_texture_coordinates must map source polygon indices to UV arrays"
        )

    required_top_level = {"version", "texture-channels", "polygons"}
    missing_top_level = required_top_level.difference(source_tube)
    if missing_top_level:
        raise ValueError(
            "source_tube is missing required fields: "
            + ", ".join(sorted(missing_top_level))
        )

    source_polygons = source_tube["polygons"]
    if not isinstance(source_polygons, list) or not source_polygons:
        raise ValueError("source_tube['polygons'] must be a non-empty list")
    if not isinstance(start_polygon_index, (int, np.integer)):
        raise TypeError("start_polygon_index must be an integer")
    start_polygon_index = int(start_polygon_index)
    if not 0 <= start_polygon_index < len(source_polygons):
        raise IndexError(
            f"start_polygon_index must be between 0 and {len(source_polygons) - 1}"
        )
    if end_polygon_index is None:
        end_polygon_index = len(source_polygons) - 1
    if not isinstance(end_polygon_index, (int, np.integer)):
        raise TypeError("end_polygon_index must be an integer or None")
    end_polygon_index = int(end_polygon_index)
    if not start_polygon_index <= end_polygon_index < len(source_polygons):
        raise IndexError(
            "end_polygon_index must be at least start_polygon_index and no greater "
            f"than {len(source_polygons) - 1}"
        )

    required_indices = range(start_polygon_index, end_polygon_index + 1)
    missing_rings = [index for index in required_indices if index not in polygon_rings_3d_m]
    if missing_rings:
        preview = ", ".join(str(index) for index in missing_rings[:10])
        suffix = "..." if len(missing_rings) > 10 else ""
        raise ValueError(
            f"missing polygon rings for {len(missing_rings)} source slices: "
            f"{preview}{suffix}"
        )
    if polygon_texture_coordinates is not None:
        missing_uvs = [
            index for index in required_indices
            if index not in polygon_texture_coordinates
        ]
        if missing_uvs:
            preview = ", ".join(str(index) for index in missing_uvs[:10])
            suffix = "..." if len(missing_uvs) > 10 else ""
            raise ValueError(
                f"missing polygon UV coordinates for {len(missing_uvs)} slices: "
                f"{preview}{suffix}"
            )

    polygon_tube = deepcopy(source_tube)
    polygon_tube["polygons"] = []

    required_polygon_fields = {
        "time",
        "center",
        "texture",
        "axes-lengths",
        "rotation",
        "points",
    }
    required_point_fields = {"x", "y", "z", "u", "v", "data"}

    for source_index in required_indices:
        source_polygon = source_polygons[source_index]
        if not isinstance(source_polygon, dict):
            raise ValueError(f"source polygon {source_index} must be a dictionary")
        missing_polygon_fields = required_polygon_fields.difference(source_polygon)
        if missing_polygon_fields:
            raise ValueError(
                f"source polygon {source_index} is missing fields: "
                + ", ".join(sorted(missing_polygon_fields))
            )

        source_points = source_polygon["points"]
        if not isinstance(source_points, list) or len(source_points) < 3:
            raise ValueError(
                f"source polygon {source_index} must contain at least three points"
            )
        for point_index, point in enumerate(source_points):
            if not isinstance(point, dict):
                raise ValueError(
                    f"source polygon {source_index} point {point_index} must be a dictionary"
                )
            missing_point_fields = required_point_fields.difference(point)
            if missing_point_fields:
                raise ValueError(
                    f"source polygon {source_index} point {point_index} is missing fields: "
                    + ", ".join(sorted(missing_point_fields))
                )

        ring = np.asarray(polygon_rings_3d_m[source_index], dtype=float)
        expected_shape = (len(source_points), 3)
        if ring.shape != expected_shape:
            raise ValueError(
                f"ring {source_index} has shape {ring.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(ring)):
            raise ValueError(f"ring {source_index} contains non-finite coordinates")

        uv_coordinates = None
        if polygon_texture_coordinates is not None:
            uv_coordinates = np.asarray(
                polygon_texture_coordinates[source_index], dtype=float
            )
            expected_uv_shape = (len(source_points), 2)
            if uv_coordinates.shape != expected_uv_shape:
                raise ValueError(
                    f"UV array {source_index} has shape {uv_coordinates.shape}; "
                    f"expected {expected_uv_shape}"
                )
            if not np.all(np.isfinite(uv_coordinates)):
                raise ValueError(
                    f"UV array {source_index} contains non-finite coordinates"
                )

        output_polygon = deepcopy(source_polygon)
        for point_index, position_m in enumerate(ring):
            output_point = output_polygon["points"][point_index]
            output_point["x"] = float(position_m[0])
            output_point["y"] = float(position_m[1])
            output_point["z"] = float(position_m[2])
            if uv_coordinates is not None:
                output_point["u"] = float(uv_coordinates[point_index, 0])
                output_point["v"] = float(uv_coordinates[point_index, 1])
        polygon_tube["polygons"].append(output_polygon)

    return polygon_tube


def writePolygonTube(
    filename: str,
    output_directory: str | os.PathLike,
    source_tube: dict,
    polygon_rings_3d_m: Mapping[int, npt.ArrayLike],
    start_polygon_index: int,
    end_polygon_index: int | None = None,
    polygon_texture_coordinates: Mapping[int, npt.ArrayLike] | None = None,
) -> str:
    """Create and write a schema-compatible polygon-only OpenSpace tube.

    Returns the path of the written JSON file.  Existing files with the same
    name are replaced, matching :func:`writeTube` behavior.
    """

    if not isinstance(filename, str) or not filename:
        raise ValueError("filename must be a non-empty string")

    polygon_tube = createPolygonTube(
        source_tube,
        polygon_rings_3d_m,
        start_polygon_index,
        end_polygon_index,
        polygon_texture_coordinates,
    )
    output_directory = os.fspath(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    filepath = os.path.join(output_directory, filename)
    print("Writing polygon tube file", filepath)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(polygon_tube, file)
    return filepath

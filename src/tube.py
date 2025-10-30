import os
import json

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
        data_dictionary["polygons"][time]["texture"] = time_polygon.textures
        

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
                    "density": time_polygon.samples[p].density
                }
            })
        data_dictionary["polygons"][time]["points"] = tube_polygon_points

        # Increment time index
        time += 1

    return data_dictionary

def writeTube(tube_filename, out_directory, time_polygons, num_ellipse_samples):
    """
    
    """
    
    # Meta data for the whole tube file
    version = [int(0), int(2)]
    texture_channels = ["density", "time-delta"]

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
    for t in range(len(texture_channels)):
        data_dictionary["texture-channels"].append(texture_channels[t])

    # Fill the dictionary with the tube data
    data_dictionary = createTube(data_dictionary, time_polygons, num_ellipse_samples)

    # Create the file
    filepath = os.path.join(out_directory, tube_filename)

    # Write the tube information to the JSON file
    with open(filepath, 'w') as fp:
        json.dump(data_dictionary, fp)
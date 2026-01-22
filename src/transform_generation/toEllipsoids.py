import os

from jinja2 import Template 

import tube_generation.util as util


ellipsoid_template = """
local coreKernels = asset.require("spice/core")
local transforms = asset.require("./{{ transforms_filename }}")

local EllipsoidRotation = {
  Type = "TimelineRotation",
  Keyframes = { {% for keyframe in keyframes %}
    ["{{ keyframe.time }}"] = {
      Type = "StaticRotation",
      Rotation = {
        {{ keyframe.x1 }}, {{ keyframe.x2 }}, {{ keyframe.x3 }},
        {{ keyframe.y1 }}, {{ keyframe.y2 }}, {{ keyframe.y3 }},
        {{ keyframe.z1 }}, {{ keyframe.z2 }}, {{ keyframe.z3 }}
      }
    }, {% endfor %}
  }
}

local EllipsoidScale = {
  Type = "TimelineScale",
  Keyframes = { {% for keyframe in keyframes %}
    ["{{ keyframe.time }}"] = {
      Type = "NonUniformStaticScale",
      Scale = {
        {{ keyframe.a }},
        {{ keyframe.b }},
        {{ keyframe.c }},
      }
    }, {% endfor %}
  }
}

local Ellipsoid = {
  Parent = transforms.TubePosition.Identifier,
  Identifier = "Ellipsoid_{{ id }}",
  Transform = {
    Scale = EllipsoidScale,
    Rotation = EllipsoidRotation
  },
  Renderable = {
    Type = "RenderableSphericalGrid"
  },
  GUI = {
    Name = "{{ name }} Ellipsoid",
    Path = "/{{ gui_path }}"
  }
}

asset.onInitialize(function()
  openspace.addSceneGraphNode(Ellipsoid)
end)

asset.onDeinitialize(function()
  openspace.removeSceneGraphNode(Ellipsoid)
end)

asset.export("Ellipsoid", Ellipsoid)

"""

def generateEllipsoids(ellipsoids_filename, output_directory, time_polygons,
                       transforms_filename, configuration):
    """
    Generate the OpenSpace asset file for the ellipsoids and write it to file.

    Input:
        ellipsoids_filename: The filename of the ellipsoids asset file to create
        output_directory: The directory to save the ellipsoids asset file to
        time_polygons: The list of TimePolygon objects containing the ellipsoid data
        transforms_filename: The filename of the transforms asset file to reference
        configuration: The configuration dictionary containing the object ID
    """

    # Create the Jinja2 keyframe data from the time_polygons
    keyframes = []
    for time_polygon in time_polygons:
        keyframe = {
            # Time
            "time": time_polygon.time,

            # Ellipse center position
            "x": time_polygon.center[0],
            "y": time_polygon.center[1],
            "z": time_polygon.center[2],

            # Rotation
            "x1": time_polygon.ellipsoid.rotation_matrix[0][0],
            "y1": time_polygon.ellipsoid.rotation_matrix[0][1],
            "z1": time_polygon.ellipsoid.rotation_matrix[0][2],
            "x2": time_polygon.ellipsoid.rotation_matrix[1][0],
            "y2": time_polygon.ellipsoid.rotation_matrix[1][1],
            "z2": time_polygon.ellipsoid.rotation_matrix[1][2],
            "x3": time_polygon.ellipsoid.rotation_matrix[2][0],
            "y3": time_polygon.ellipsoid.rotation_matrix[2][1],
            "z3": time_polygon.ellipsoid.rotation_matrix[2][2],

            # Scale
            "a": time_polygon.ellipsoid.axes_lengths[0],
            "b": time_polygon.ellipsoid.axes_lengths[1],
            "c": time_polygon.ellipsoid.axes_lengths[2]
        }
        keyframes.append(keyframe)

    # The OpenSpace identifiers cannot contain any spaces or special characters
    object_id = configuration["object_id"]
    object_identifier = object_id.replace(" ", "_")

    # Specify the data entries in the Jinja2 template above
    data = {
        "id": object_identifier, 
        "name": object_id,
        "gui_path": "B612",
        "transforms_filename": transforms_filename,
        "keyframes": keyframes
    }

    # Run Jinja2 to generate a valid OpenSpace asset using the template
    ellipsoid_asset_template = Template(ellipsoid_template)
    ellipsoid_asset = ellipsoid_asset_template.render(data)
    #print(ellipsoid_asset)

    # Save it to file
    filepath = os.path.join(output_directory, ellipsoids_filename)
    util.writeFile(filepath, ellipsoid_asset)

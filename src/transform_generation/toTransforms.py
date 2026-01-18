import os

from jinja2 import Template 

import tube_generation.util as util

transforms_template = """
local coreKernels = asset.require("spice/core")
local sunTransforms = asset.require("scene/solarsystem/sun/transforms")

local TubeParent = {
  Identifier = "Parent_{{ id }}",
  Parent = sunTransforms.SunCenter.Identifier,
  Transform = {
    Rotation = {
      Type = "SpiceRotation",
      SourceFrame = coreKernels.Frame.EclipJ2000,
      DestinationFrame = coreKernels.Frame.Galactic
    }
  },
  GUI = {
    Name = "{{ name }} Parent",
    Path = "/{{ gui_path }}",
    Hidden = true
  }
}

local TubePosition = {
  Identifier = "Position_{{ id }}",
  Parent = TubeParent.Identifier,
  Transform = {
    Translation = {
      Type = "TimelineTranslation",
      Keyframes = { {% for keyframe in keyframes %}
        ["{{ keyframe.time }}"] = {
          Type = "StaticTranslation",
          Position = {
            {{ keyframe.x }},
            {{ keyframe.y }},
            {{ keyframe.z }}
          }
        }, {% endfor %}
      }
    }
  },
  GUI = {
    Name = "{{ name }} Position",
    Path = "/{{ gui_path }}"
  }
}

asset.onInitialize(function()
  openspace.addSceneGraphNode(TubeParent)
  openspace.addSceneGraphNode(TubePosition)
end)

asset.onDeinitialize(function()
  openspace.removeSceneGraphNode(TubePosition)
  openspace.removeSceneGraphNode(TubeParent)
end)

asset.export("TubeParent", TubeParent)
asset.export("TubePosition", TubePosition)

"""

def generateTransforms(transforms_filename, output_directory, time_polygons,
                       configuration):
    """
    Generate the OpenSpace asset file for the transforms of the tube position over time.

    Input:
        transforms_filename: The filename for the transforms asset file to create
        output_directory: The directory to save the new transforms asset file to
        time_polygons: The TubePolygons that create the tube
        configuration: The configuration settings from the user
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
        "keyframes": keyframes
    }

    # Run Jinja2 to generate a valid OpenSpace asset using the template
    transform_asset_template = Template(transforms_template)
    transforms_asset = transform_asset_template.render(data)
    #print(transforms_asset)

    # Save it to file
    filepath = os.path.join(output_directory, transforms_filename)
    util.writeFile(filepath, transforms_asset)

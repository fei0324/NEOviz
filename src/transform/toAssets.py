from jinja2 import Template 
import sys
from jinjaData.data_dummy import keyframes # Replace with input folder and filename

# Requires pip install Jinja2

# This script takes in raw jinja2 data from another python file, which is used to fill in
# the templates below. Jinja then writes OpenSpace asset files using the templates.
#
# Specify input above and output in the end of the file.

transformsTemplate = """
local coreKernels = asset.require("spice/core")
local sunTransforms = asset.require("scene/solarsystem/sun/transforms")

local TubeParent = {
  Identifier = "Parent{{ id }}",
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
  Identifier = "Position{{ id }}",
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


ellipsoidTemplate = """
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
  Identifier = "Ellipsoid{{ id }}",
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

def writeAssetFile(filename, asset):
  file = open(filename, "w")
  file.write(asset)
  file.close()

def main():
  transformsOutputFilename = "./transforms/transforms_dummy.asset"
  ellipsoidOutputFilename = "./ellipsoids/ellipsoids_dummy.asset"

  # Data for the template, replace the items on the right to whatever you want
  data = {
    "id": "Dummy_Test", # Cannot contain spaces or any special characters
    "name": "Dummy Test",
    "gui_path": "Test",
    "transforms_filename": "transforms_dummy",
    "keyframes": keyframes
  }

  transform_asset_template = Template(transformsTemplate)
  ellipsoid_asset_template = Template(ellipsoidTemplate)

  transforms_asset = transform_asset_template.render(data)
  #print(transforms_asset)
  writeAssetFile(transformsOutputFilename, transforms_asset)

  ellipsoid_asset = ellipsoid_asset_template.render(data)
  #print(ellipsoid_asset)
  writeAssetFile(ellipsoidOutputFilename, ellipsoid_asset)

main()

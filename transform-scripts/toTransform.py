from jinja2 import Template # pip install Jinja2
import sys
from transformsData.filename import keyframes # Replace with input filename

# This script takes in data from another python file that contains raw data for jinja2
# that data is then used to fill in this template below. Jinja then creates an asset
# file according to the template.
# Specify input above and output in the end of the file.

template = """
local coreKernels = asset.require("spice/core")
local transforms = asset.require("scene/solarsystem/sun/transforms")

local TubeParent = {
  Identifier = "Parent{{ id }}",
  Parent = transforms.SunCenter.Identifier,
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
            {{ keyframe.xPos }},
            {{ keyframe.yPos }},
            {{ keyframe.zPos }}
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

def writeAssetFile(filename, asset):
  file = open(filename, "w")
  file.write(asset)
  file.close()

def main():
  outputFileName = "./transforms/filename.asset"

  # Data for the template, replace the items on the right to whatever you want
  data = {
    "id": "Identifier", # Cannot contain spaces or any special characters
    "name": "Name",
    "gui_path": "Path",
    "keyframes": keyframes
  }

  asset_template = Template(template)
  asset_test = asset_template.render(data)
  #print(asset_test)

  writeAssetFile(outputFileName, asset_test)

main()

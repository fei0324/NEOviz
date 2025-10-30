import json

# This script takes a tube data json file and converts it into a python file that
# contains raw information for jinja2 to automatically create OpenSpace asset files.
# These asset files can then be created with the python script "toAssets.py".
#
# Specify input and output in the end of the file.

def readFile(filename):
  file = open(filename, 'r')
  content = file.read()
  return content

def createJinjaData(fileContent):
  jsonData = json.loads(fileContent)
  data = "keyframes = [\n"

  for i in range(0, len(jsonData["polygons"])):
    polygon = jsonData["polygons"][i]
    isLast = i == len(jsonData["polygons"]) - 1

    data = data + "  {\n"
    data = data + "    \"time\": \"" + str(polygon["time"]) + "\",\n"

    # Position
    data = data + "    \"x\": \"" + str(polygon["center"]["x"]) + "\",\n"
    data = data + "    \"y\": \"" + str(polygon["center"]["y"]) + "\",\n"
    data = data + "    \"z\": \"" + str(polygon["center"]["z"]) + "\",\n"

    # Rotation
    data = data + "    \"x1\": \"" + str(polygon["rotation"]["x1"]) + "\",\n"
    data = data + "    \"x2\": \"" + str(polygon["rotation"]["x2"]) + "\",\n"
    data = data + "    \"x3\": \"" + str(polygon["rotation"]["x3"]) + "\",\n"
    data = data + "    \"y1\": \"" + str(polygon["rotation"]["y1"]) + "\",\n"
    data = data + "    \"y2\": \"" + str(polygon["rotation"]["y2"]) + "\",\n"
    data = data + "    \"y3\": \"" + str(polygon["rotation"]["y3"]) + "\",\n"
    data = data + "    \"z1\": \"" + str(polygon["rotation"]["z1"]) + "\",\n"
    data = data + "    \"z2\": \"" + str(polygon["rotation"]["z2"]) + "\",\n"
    data = data + "    \"z3\": \"" + str(polygon["rotation"]["z3"]) + "\",\n"

    # Scale
    data = data + "    \"a\": \"" + str(polygon["axes-lengths"]["a"]) + "\",\n"
    data = data + "    \"b\": \"" + str(polygon["axes-lengths"]["b"]) + "\",\n"
    data = data + "    \"c\": \"" + str(polygon["axes-lengths"]["c"]) + "\"\n"

    if isLast:
      data = data + "  }\n"
    else:
      data = data + "  },\n"

  data = data + "]\n"
  return data

def writeDataFile(filename, data):
  file = open(filename, "w")
  file.write(data)
  file.close()

def main():
  # Specify the input and output
  inputFilename = "./jsonData/tube_2023_CX1.json"
  outputFilename ="./jinjaData/jinja_2023_CX1.py"

  content = readFile(inputFilename)
  data = createJinjaData(content)
  #print(data)

  writeDataFile(outputFilename, data)

main()

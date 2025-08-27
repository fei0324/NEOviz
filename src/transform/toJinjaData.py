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
    data = data + "    \"x1\": \"" + str(polygon["axes-direction"]["x1"]) + "\",\n"
    data = data + "    \"x2\": \"" + str(polygon["axes-direction"]["x2"]) + "\",\n"
    data = data + "    \"x3\": \"" + str(polygon["axes-direction"]["x3"]) + "\",\n"
    data = data + "    \"y1\": \"" + str(polygon["axes-direction"]["y1"]) + "\",\n"
    data = data + "    \"y2\": \"" + str(polygon["axes-direction"]["y2"]) + "\",\n"
    data = data + "    \"y3\": \"" + str(polygon["axes-direction"]["y3"]) + "\",\n"
    data = data + "    \"z1\": \"" + str(polygon["axes-direction"]["z1"]) + "\",\n"
    data = data + "    \"z2\": \"" + str(polygon["axes-direction"]["z2"]) + "\",\n"
    data = data + "    \"z3\": \"" + str(polygon["axes-direction"]["z3"]) + "\",\n"

    # Scale
    data = data + "    \"a\": \"" + str(polygon["axes-length"]["a"]) + "\",\n"
    data = data + "    \"b\": \"" + str(polygon["axes-length"]["b"]) + "\",\n"
    data = data + "    \"c\": \"" + str(polygon["axes-length"]["c"]) + "\"\n"

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
  inputFilename = "./jsonData/dummy-data.json"
  outputFilename ="./jinjaData/data_dummy.py"

  content = readFile(inputFilename)
  data = createJinjaData(content)
  #print(data)

  writeDataFile(outputFilename, data)

main()

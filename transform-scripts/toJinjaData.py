import json

# This script takes a tube data json file and converts it into a python
# file that contains raw information for jinja2 to automatically create an
# OpenSpace asset file. This asset file can then be created with the python script
# "toTransform.py".
# Specify input and output in the end of the file.

def readFile(filename):
  file = open(filename, 'r')
  content = file.read()
  return content

def createJinjaData(fileContent):
  jsonData = json.loads(fileContent)
  data = "keyframes = [\n"

  isFirst = True
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
    data = data + "    \"a11\": \"" + str(polygon["rotation"][0]) + "\",\n"
    data = data + "    \"a12\": \"" + str(polygon["rotation"][1]) + "\",\n"
    data = data + "    \"a13\": \"" + str(polygon["rotation"][2]) + "\",\n"
    data = data + "    \"a21\": \"" + str(polygon["rotation"][3]) + "\",\n"
    data = data + "    \"a22\": \"" + str(polygon["rotation"][4]) + "\",\n"
    data = data + "    \"a23\": \"" + str(polygon["rotation"][5]) + "\",\n"
    data = data + "    \"a31\": \"" + str(polygon["rotation"][6]) + "\",\n"
    data = data + "    \"a32\": \"" + str(polygon["rotation"][7]) + "\",\n"
    data = data + "    \"a33\": \"" + str(polygon["rotation"][8]) + "\",\n"

    # Scale
    data = data + "    \"a\": \"" + str(polygon["scale"]["a"]) + "\",\n"
    data = data + "    \"b\": \"" + str(polygon["scale"]["b"]) + "\",\n"
    data = data + "    \"c\": \"" + str(polygon["scale"]["c"]) + "\"\n"

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
  inputFilename = "jsonData/dummy-data.json"
  outputFilename ="./jinjaData/data_dummy.py"

  content = readFile(inputFilename)
  data = createJinjaData(content)
  #print(data)

  writeDataFile(outputFilename, data)

main()

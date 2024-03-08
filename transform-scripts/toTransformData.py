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

def createTransformData(fileContent):
  data = "keyframes = [\n"
  isFirst = True

  jsonData = json.loads(fileContent)

  for i in range(0, len(jsonData["polygons"])):
    polygon = jsonData["polygons"][i]
    isLast = i == len(jsonData["polygons"]) - 1

    data = data + "  {\n"
    data = data + "    \"time\": \"" + str(polygon["time"]) + "\",\n"
    data = data + "    \"xPos\": \"" + str(polygon["center"]["x"]) + "\",\n"
    data = data + "    \"yPos\": \"" + str(polygon["center"]["y"]) + "\",\n"
    data = data + "    \"zPos\": \"" + str(polygon["center"]["z"]) + "\"\n"

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
  inputFile = "jsonData/filename.json"
  outputFileName ="./transformsData/filename.py"

  content = readFile(inputFile)
  data = createTransformData(content)
  #print(data)

  writeDataFile(outputFileName, data)

main()

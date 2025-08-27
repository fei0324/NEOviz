from decimal import *
import sys

def readHrz(filename):
  file = open(filename, 'r')
  lines = file.readlines()

  # Strip new line character
  for line in lines:
    line = line.strip()
  return lines

# For Debug
def printLines(lines):
  for line in lines:
    print(line)

# Filter out only the lines that contain data
def getOnlyDataLines(lines):
  onlyDataLines = []
  dataStarted = False
  for i in range(0, len(lines)):
    if (lines[i][0] == '$' and not dataStarted):
      dataStarted = True
    elif (lines[i][0] == '$'):
      break
    elif (dataStarted):
      onlyDataLines.append(lines[i])
  return onlyDataLines

# Show a progress bar in the console
def progressBar(countValue, total, suffix=''):
  barLength = 50
  filledUpLength = int(round(barLength * countValue / float(total)))
  percentage = round(100.0 * countValue/float(total), 1)
  bar = '=' * filledUpLength + '-' * (barLength - filledUpLength)
  sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
  sys.stdout.flush()

# Write the tube data file
def createTubeFile(datalines):
  tube = """
  {
    "version": {
      "major": 0,
      "minor": 1
    },
    "polygons": [\n"""
  isDateLine = True
  isFirst = True
  for i in range(0, len(datalines)):
    progressBar(i, len(datalines))
    words = datalines[i].split()

    # Horizons Vector format is used, lines alternate between timestamps and positions
    # Timestamp
    if isDateLine:
      if not isFirst:
        tube = tube + "      },\n"
      tube = tube + "      {\n        \"time\": \""
      tube = tube + words[3]
      tube = tube + " "
      tube = tube + words[4]
      tube = tube + "\",\n"
      isDateLine = False
    # Position
    else:
      # The amount to offset from the horizons trajectory to create the tube,
      # artificially add "uncertainty"
      offset = 10000000
      words = datalines[i].split()

      # Convert to meters form Km
      xPos = Decimal(words[0]) * 1000
      yPos = Decimal(words[1]) * 1000
      zPos = Decimal(words[2]) * 1000

      tube = tube + "        \"center\": {\n"
      tube = tube + "          \"x\": " + str(xPos) + ",\n"
      tube = tube + "          \"y\": " + str(yPos) + ",\n"
      tube = tube + "          \"z\": " + str(zPos) + "\n"
      tube = tube + "        },\n"

      tube = tube + "        \"points\": [\n"
      tube = tube + "          {\n"
      tube = tube + "            \"x\": " + str(xPos + offset) + ",\n"
      tube = tube + "            \"y\": " + str(yPos + offset) + ",\n"
      tube = tube + "            \"z\": " + str(zPos) + "\n"
      tube = tube + "          },\n"
      tube = tube + "          {\n"
      tube = tube + "            \"x\": " + str(xPos - offset) + ",\n"
      tube = tube + "            \"y\": " + str(yPos + offset) + ",\n"
      tube = tube + "            \"z\": " + str(zPos) + "\n"
      tube = tube + "          },\n"
      tube = tube + "          {\n"
      tube = tube + "            \"x\": " + str(xPos - offset) + ",\n"
      tube = tube + "            \"y\": " + str(yPos - offset) + ",\n"
      tube = tube + "            \"z\": " + str(zPos) + "\n"
      tube = tube + "          },\n"
      tube = tube + "          {\n"
      tube = tube + "            \"x\": " + str(xPos + offset) + ",\n"
      tube = tube + "            \"y\": " + str(yPos - offset) + ",\n"
      tube = tube + "            \"z\": " + str(zPos) + "\n"
      tube = tube + "          }\n"
      tube = tube + "        ]\n"

      isDateLine = True
    if isFirst:
      isFirst = False

  tube = tube + """
      }
    ]
  }"""
  return tube

def writeTubeFile(filename, tube):
  file = open(filename, "w")
  file.write(tube)
  file.close()

def main():
  lines = readHrz("./hrzData/filename.hrz")
  #printLines(lines)

  onlyDataLines = getOnlyDataLines(lines)
  #printLines(onlyDataLines)

  tube = createTubeFile(onlyDataLines)
  #print(tube)

  writeTubeFile("./jsonData/filename.json", tube)

main()

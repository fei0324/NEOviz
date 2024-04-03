import os
import sys
import math
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm
import time
import spiceypy as spice

# To initialize spice
METAKERNEL = "meta-kernel.tm"

# Offset spice Id to not collide with any existing NAIF id
ID_OFFSET = 1000000

# Image size
IMAGE_WIDTH = 5400
IMAGE_HEIGHT = 2700

# Set the reference distance to Earth
EARTH_RADIUS = 6357 # minimum radius in Km
#EARTH_RADIUS = 39000 # minimum orbital insertion distance for Earth and 2004 MN4


# Show a progress bar in the console
def progressBar(count_value, total, suffix=''):
  bar_length = 50
  filled_up_Length = int(round(bar_length * count_value / float(total)))
  percentage = round(100.0 * count_value/float(total), 1)
  bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
  sys.stdout.write('[%s] %s%s %s/%s...%s\r' %(bar, percentage, '%', str(count_value), str(total), suffix))
  sys.stdout.flush()


# Function to generate the impact map images
def raw_impact_map(
    input_path,
    kernel_list,
    timeStart,
    timeEnd,
    nightPixels,
    brushSize,
    cityFilename,
    impactFilename,
    impactorsList
  ):

  # Create images that we can draw on
  cityImage = Image.new("RGBA", (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0, 0))
  impactImage = Image.new("RGBA", (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0, 0))
  drawCityImage = ImageDraw.Draw(cityImage)
  drawImpactImage = ImageDraw.Draw(impactImage)

  # Different brushes
  halfBrush = int(brushSize/2)
  quarterBrush = int(brushSize/4)

  # Specify the time range to search in
  timerange = spice.cell_double(200)
  spice.wninsd(timeStart, timeEnd, timerange)

  # Loop over the variants
  variant_id = ID_OFFSET
  impact = False
  print("Processing variants...")
  sys.stdout.flush()
  for variant in kernel_list:
    variant_name = str(variant_id)

    # Make sure it is a kernel file and not a directory
    kernelFile = os.path.join(input_path, variant)
    if not os.path.isfile(kernelFile) or not os.path.exists(kernelFile):
      print(kernelFile, "is not a file or could not be found")
      sys.stdout.flush()
      variant_id += 1
      continue

    # Load the kernel file for this variant
    spice.furnsh(kernelFile)

    # Get times when the variant is close enough to Earth
    try:
      result = spice.gfdist(
        variant_name,       # Name of the target body
        "NONE",             # Aberration correction flag
        "EARTH",            # Name of the observing body
        "<",                # Relational operator (example <, = or >)
        EARTH_RADIUS,       # Reference value
        0.0,                # Adjustment value for absolute extrema searches
        spice.spd(),        # Step size used for locating extrema and roots (The number of seconds in a day)
        100,                # Workspace window interval count (from example)
        timerange           # SPICE window to which the search is confined (from example)
      )
    except:
      print("Missing kernel file", variant_id - ID_OFFSET + 1)
      sys.stdout.flush()
      spice.reset()
      variant_id += 2
      spice.unload(kernelFile)
      continue

    # Check that we got a valid result
    if spice.wncard(result) == 0:
      #print("Variant", variant_name, "does not impact Earth")
      #sys.stdout.flush()
      variant_id += 1
      spice.unload(kernelFile)
      continue

    # Get results
    for i in range(spice.wncard(result)):
      # Get time range when object is within the reference distance
      [start, stop] = spice.wnfetd(result, i)
      impactTime = spice.timout(
        start,                      # Epoch in seconds past the ephemeris epoch J2000
        "YYYY MON DD HR:MN:SC.###", # Format for the output string
        41                          # Length of the output string plus 1
      )

      # Get cartesian X, Y, Z position of object at the impact time
      # TODO maybe change this reference frame to IAU_EARTH to make sure we always follow rotation?
      [position, lt] = spice.spkpos(
        variant_name,     # Target body name to check position for
        start,            # Time for the position in seconds
        "IAU_EARTH",      # Reference frame of output position vector
        "NONE",           # Aberration correction flag
        "EARTH"           # Observing body name, reference frame of output position
      )

      print("Variant", variant_name, "impacts Earth on", impactTime)
      #print("position", position)
      #sys.stdout.flush()
      impactorsList.append(variant_name)
      impact = True

      # Get lat long coordinate
      [alt, long, lat] = spice.reclat(position)
      latDeg = math.degrees(lat)
      longDeg = math.degrees(long)
      print("coord:", latDeg, longDeg)
      #sys.stdout.flush()

      # Flip the y axis for the image, north is up
      latDeg *= -1

      # Normalize coordinate to be within 0 to 1 range
      pixelLat = (latDeg + 90)/180
      pixelLong = (longDeg + 180)/360

      # Find corresponding pixel in equirectangular texture
      pixelY = round(pixelLat * IMAGE_HEIGHT)
      pixelX = round(pixelLong * IMAGE_WIDTH)
      print("pixel", pixelX, pixelY)
      sys.stdout.flush()

      # Get color of night image, sample an area around the pixel and add up the color
      # +1 in loop since range is [a, b[
      nightValue = 0
      for j in range(-halfBrush, halfBrush + 1):
        nightX = pixelX + j
        if (nightX < 0 or nightX >= IMAGE_WIDTH):
            # Dont cross borders of the image, early out for performance
            continue

        for k in range(-halfBrush, halfBrush + 1):
          nightY = pixelY + k
          if (nightY < 0 or nightY >= IMAGE_HEIGHT):
            # Dont cross borders of the image
            continue

          # TODO: Add a circular shape of the sampling not a sqare shape and distance based weights

          # Only use the red channel in the night layer
          nightValue += nightPixels[nightX, nightY][0]

      # Normaize the color
      nightValue = round(nightValue / (brushSize * brushSize))
      nightValue *= 1.8
      nightValue = round(np.clip(round(nightValue), 0, 255))
      #print("resulting night value", nightValue)
      #sys.stdout.flush()

      # Draw dot
      shape = [(pixelX - halfBrush, pixelY - halfBrush), (pixelX + halfBrush, pixelY + halfBrush)]
      drawCityImage.ellipse(shape, fill=(nightValue, nightValue, nightValue, 255))
      drawImpactImage.ellipse(shape, fill=(200, 200, 200, 255))

    # Reset
    spice.unload(kernelFile)
    variant_id += 1

  # Check if there were any impacts
  if not impact:
    print("No variant impacts Earth!")
    sys.stdout.flush()
    sys.exit()

  # Make it all blurry
  blurryCityImage = drawCityImage._image.filter(ImageFilter.GaussianBlur(radius = halfBrush))
  blurryImpactImage = drawImpactImage._image.filter(ImageFilter.GaussianBlur(radius = halfBrush))

  # Save the resulting images
  blurryCityImage.save(cityFilename)
  blurryImpactImage.save(impactFilename)


# Apply transfer function
def color_impact_map(
    imagePixels,
    filename,
    transferFunction
  ):

  # Create new images to put color to
  newImage = Image.new("RGBA", (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0, 0))
  drawImage = ImageDraw.Draw(newImage)

  # Apply color to each pixel in the image
  print("Applying color...")
  sys.stdout.flush()
  for j in range(IMAGE_WIDTH):
    for k in range(IMAGE_HEIGHT):
      color = imagePixels[j, k]

      # If fully transparent then there is no need to paint this pixel
      if color[3] < 1:
        continue

      transformedColor = transferFunction.to_rgba(color[0])
      finalColor = (round(transformedColor[0]*255), round(transformedColor[1]*255), round(transformedColor[2]*255), color[3])

      drawImage.point((j, k), fill=(finalColor))

  # Save the resulting images
  drawImage._image.save(filename)


if __name__ == "__main__":
  start_time = time.time()

  # Initialize SPICE
  spice.furnsh(METAKERNEL)

  # Choose what data to look at
  # 0 = 2023 CX1, 1 = 2004 MN4, 2 = 2012 DA14
  dataItem = 1

  # Get the kernel files
  dataList = ["./data/2023 CX1/openspace_variants/", "./data/2004 MN4/openspace_variants_high_ip/", "./data/2012 DA14/2012_03_05T06_24_12/openspace_variants/"]
  input_path = dataList[dataItem]
  kernel_list = os.listdir(input_path)
  kernel_list.sort()

  # Include night layer image to asses population density
  directory = os.path.dirname(__file__)
  nightFile = os.path.join(directory, "earth_night.png")
  nightImage = Image.open(nightFile)
  nightPixels = nightImage.load()

  # Time range
  # 2023 CX1: start "2023-02-13T02:39:00.000", end "2023-02-13T03:40:00.000"
  # 2004 MN4: start "2029-02-01", end "2029-08-17"
  # 2012 DA14: start "2012-03-06", end "2013-09-30"
  dataStartList = ["2023-02-13T02:39:00.000", "2029-02-01", "2012-03-06"]
  dataEndList = ["2023-02-13T03:39:00.000", "2029-08-17", "2013-09-30"]
  timeStart = spice.str2et(dataStartList[dataItem])
  timeEnd = spice.str2et(dataEndList[dataItem])

  # Filenames
  dataNames = ["2023-CX1", "2004-MN4", "2012-DA14"]
  extension = ".png"
  cityFilenameStart = "images/city-" + dataNames[dataItem]
  impactFilenameStart = "images/impact-" + dataNames[dataItem]
  colorCityFilenameStart = "images/city-color-" + dataNames[dataItem]
  colorImpactFilenameStart = "images/impact-color-" + dataNames[dataItem]

  # Settings
  brushSize = 20
  maxValueCity = 15
  maxValueImpact = 100

  cityFilename = cityFilenameStart + "-BS_" + str(brushSize) + extension
  impactFilename = impactFilenameStart + "-BS_" + str(brushSize) + extension

  # Get the raw map
  impactors = []
  raw_impact_map(
    input_path,
    kernel_list,
    timeStart,
    timeEnd,
    nightPixels,
    brushSize,
    cityFilename,
    impactFilename,
    impactors
  )

  #print("Impactors:")
  #for impactor in impactors:
  #  print(impactor)
  #  sys.stdout.flush()

  # Load the images that were just created again
  cityFile = os.path.join(directory, cityFilename)
  cityImage = Image.open(cityFile)
  cityImagePixels = cityImage.load()

  impactFile = os.path.join(directory, impactFilename)
  impactImage = Image.open(impactFile)
  impactImagePixels = impactImage.load()

  # Colormap settings
  colorMaps = [cm.viridis, cm.plasma, cm.inferno, cm.magma, cm.cividis]
  colorMapsNames = ["viridis", "plasma", "inferno", "magma", "cividis"]
  colorMapIndex = 2

  colorCityFilename = colorCityFilenameStart + "-BS_" + str(brushSize) + "_CV_" + str(maxValueCity) + "_" + colorMapsNames[colorMapIndex] + extension
  colorImpactFilename = colorImpactFilenameStart + "-BS_" + str(brushSize) + "_IV_" + str(maxValueImpact) + "_" + colorMapsNames[colorMapIndex] + extension

  # Normalize to the transfer funciton range
  normalization = mpl.colors.Normalize(vmin=0, vmax=maxValueCity)
  cityMap = cm.ScalarMappable(norm=normalization, cmap=colorMaps[colorMapIndex])

  normalization = mpl.colors.Normalize(vmin=0, vmax=maxValueImpact)
  impactMap = cm.ScalarMappable(norm=normalization, cmap=colorMaps[colorMapIndex])

  # Apply a tranfer function to the map to get color
  color_impact_map(
    cityImagePixels,
    colorCityFilename,
    cityMap
  )

  # Apply a tranfer function to the map to get color
  color_impact_map(
    impactImagePixels,
    colorImpactFilename,
    impactMap
  )

  print("--- %s seconds ---" % (time.time() - start_time))
  sys.stdout.flush()


# Impactors 2004 MN4: 88, 218, 372, 426, 498, 733, 738, 881, 893, 949, 993

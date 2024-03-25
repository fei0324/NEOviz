import os
import sys
import math
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm
import time
import spiceypy as spice


# List of valid Relational operators:
# "=", "<", ">", "LOCMIN", "ABSMIN", "ABSMIN", "LOCMAX", "ABSMAX", "ABSMAX"

# Offset spice Id to not collide with any existing NAIF id
ID_OFFSET = 1000000

# Set the reference distance to Earth
EARTH_RADIUS = 6357 # minimum radius in Km


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
    nVariants,
    kernel_list,
    timeStart,
    timeEnd,
    imageWidth,
    imageHeight,
    nightPixels,
    brushSize,
    cityFilename,
    impactFilename
  ):

  # Set up the images
  cityImage = Image.new("RGBA", (imageWidth, imageHeight), (0, 0, 0, 0))
  impactImage = Image.new("RGBA", (imageWidth, imageHeight), (0, 0, 0, 0))

  # Convert to images we can draw on
  drawCityImage = ImageDraw.Draw(cityImage)
  drawImpactImage = ImageDraw.Draw(impactImage)
  halfBrush = int(brushSize/2)
  quarterBrush = int(brushSize/4)

  # Specify the time range to search in
  timerange = spice.cell_double(200)
  spice.wninsd(timeStart, timeEnd, timerange)

  # Loop over the variants
  variant_id = ID_OFFSET
  variantCounter = 0
  for variant in kernel_list:
    variant_name = str(variant_id)

    # Show progress
    progressBar(variantCounter + 1, nVariants, "Processing variants")

    # Get times when the variant is close enough to Earth
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

    # Check that we got a valid result
    if spice.wncard(result) == 0:
      #print("Variant", variant_name, "does not impact Earth")
      variant_id += 1
      variantCounter += 1
      continue

    # Get results
    prevExist = False
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

      # Get the distance, should be close to the reference distance set
      distance = spice.vnorm(position)

      print("Variant", variant_name, "impacts Earth on", impactTime, "distance from center:", distance)
      print("position", position)
      hit = True

      # Get lat long coordinate
      [alt, long, lat] = spice.reclat(position)
      latDeg = math.degrees(lat)
      longDeg = math.degrees(long)
      print("coord:", latDeg, longDeg)

      # Flip the y axis for the image, north is up
      latDeg *= -1

      # Normalize coordinate to be within 0 to 1 range
      pixelLat = (latDeg + 90)/180
      pixelLong = (longDeg + 180)/360

      # Find corresponding pixel in equirectangular texture
      pixelY = round(pixelLat * imageHeight)
      pixelX = round(pixelLong * imageWidth)
      #print("pixel", pixelX, pixelY)

      # Get color of night image, sample an area around the pixel and add up the color
      # +1 in loop since range is [a, b[
      nightValue = 0
      for j in range(-halfBrush, halfBrush + 1):
        nightX = pixelX + j
        if (nightX < 0 or nightX >= imageWidth):
            # Dont cross borders of the image, early out for performance
            continue

        for k in range(-halfBrush, halfBrush + 1):
          nightY = pixelY + k
          if (nightY < 0 or nightY >= imageHeight):
            # Dont cross borders of the image
            continue

          # TODO: Add a circular shape of the sampeling not a sqare shape and distance based weights

          # Only use the red channel in the night layer
          pixedlcoord = nightX, nightY
          nightValue += nightPixels[pixedlcoord][0]

      # Normaize the color
      nightValue = round(nightValue / (brushSize * brushSize))
      nightValue *= 1.5  # Make it a little brighter
      nightValue = np.clip(round(nightValue), 0, 255) # Clip color if too large values
      #print("resulting night value", nightValue)

      # Draw dot
      shape = [(int(pixelX) - halfBrush, pixelY - halfBrush), (pixelX + halfBrush, pixelY + halfBrush)]
      drawCityImage.ellipse(shape, fill=(nightValue, nightValue, nightValue, 255))
      drawImpactImage.ellipse(shape, fill=(100, 100, 100, 255))

      # Make it blurry
      blurryCityImage = drawCityImage._image.filter(ImageFilter.GaussianBlur(radius = quarterBrush))
      blurryImpactImage = drawImpactImage._image.filter(ImageFilter.GaussianBlur(radius = quarterBrush))

      # Find image from previous sample(s) and add them together
      if prevExist:
        cityPixels = []
        impactPixels = []
        for j in range(imageWidth):
          for k in range(imageHeight):
            # City
            prevCityColor = cityImage.getpixel((j, k))
            currCityColor = blurryCityImage.getpixel((j, k))

            cityRed = round(prevCityColor[0] + currCityColor[0])
            cityGreen = round(prevCityColor[1] + currCityColor[1])
            cityBlue = round(prevCityColor[2] + currCityColor[2])
            cityAlpha = round(prevCityColor[3] + currCityColor[3])

            cityPixels.append((cityRed, cityGreen, cityBlue, cityAlpha))

            # Impact
            prevImpactColor = impactImage[j, k]
            currImpactColor = blurryImpactImage[j, k]

            impactRed = round(prevImpactColor[0] + currImpactColor[0])
            impactGreen = round(prevImpactColor[1] + currImpactColor[1])
            impactBlue = round(prevImpactColor[2] + currImpactColor[2])
            impactAlpha = round(prevImpactColor[3] + currImpactColor[3])

            impactPixels.append((impactRed, impactGreen, impactBlue, impactAlpha))

        cityImage = Image.fromarray(cityPixels)
        impactImage = Image.fromarray(impactPixels)
      else:
        prevExist = True
        cityImage = blurryCityImage
        impactImage = blurryImpactImage

    # Reset
    variant_id += 1
    variantCounter += 1

  # Save the resulting images
  cityImage.save(cityFilename)
  impactImage.save(impactFilename)


# Apply transfer function
def color_impact_map(
    cityImage,
    impactImage,
    cityImagePixels,
    impactImagePixels,
    imageWidth,
    imageHeight,
    cityFilename,
    impactFilename,
    maxValue,
    colorMap
  ):

  # Normalize to the transfer funciton range
  normalization = mpl.colors.Normalize(vmin=0, vmax=maxValue)
  cityMap = cm.ScalarMappable(norm=normalization, cmap=colorMap)

  normalization = mpl.colors.Normalize(vmin=0, vmax=255)
  impactMap = cm.ScalarMappable(norm=normalization, cmap=colorMap)

  # Create new images to put color to
  newCityImage = Image.new("RGBA", (imageWidth, imageHeight), (0, 0, 0, 0))
  newImpactImage = Image.new("RGBA", (imageWidth, imageHeight), (0, 0, 0, 0))
  drawCityImage = ImageDraw.Draw(cityImage)
  drawImpactImage = ImageDraw.Draw(impactImage)

  pixelCounter = 0
  for j in range(imageWidth):
    for k in range(imageHeight):
      progressBar(pixelCounter + 1, imageWidth * imageHeight, "Applying color")

      # City
      cityColor = cityImagePixels[j, k]
      transformedCityColor = cityMap.to_rgba(cityColor[0])
      #print("City: Value", cityColor[0], "tranformed to color", transformedCityColor)
      finalCityColor = (round(transformedCityColor[0]*255), round(transformedCityColor[1]*255), round(transformedCityColor[2]*255), cityColor[3])

      drawCityImage.point((j, k), fill=(finalCityColor))

      # Impact
      impactColor = impactImagePixels[j, k]
      transformedImpactColor = impactMap.to_rgba(impactColor[0])
      #print("Impact: Value", impactValue, "tranformed to color", transformedImpactColor)
      #print("\n")
      finalImpactColor = (round(transformedImpactColor[0]*255), round(transformedImpactColor[1]*255), round(transformedImpactColor[2]*255), impactColor[3])

      drawImpactImage.point((j, k), fill=(finalImpactColor))
      pixelCounter += 1

  # Save the resulting images
  drawCityImage._image.save("impact_city_color.png")
  drawImpactImage._image.save("impact_color.png")


if __name__ == "__main__":
  start_time = time.time()

  # Initialize SPICE
  METAKERNEL = 'meta-kernel.tm'
  spice.furnsh(METAKERNEL)

  # Get the kernel files
  input_path = "./data/2012 DA14/2012_03_05T06_24_12/openspace_variants/"
  kernel_list = os.listdir(input_path)
  kernel_list.sort()

  # Load all kernels
  noneLoaded = True
  nVariants = 0
  for kernel in kernel_list:
    progressBar(nVariants + 1, len(kernel_list), "Loading kernels")

    # Make sure it is a kernel file and not a directory
    kernelFile = os.path.join(input_path, kernel)
    if not os.path.isfile(kernelFile):
      print(kernelFile, "is not a file")
      continue

    # Load the kernel file for this variant
    spice.furnsh(kernelFile)
    noneLoaded = False
    nVariants += 1

  if noneLoaded:
    print("No kernels were loaded")
    sys.exit()
  else:
    print(nVariants, "variants loaded")

  # Settings
  imageWidth = 5400
  imageHeight = 2700

  # Filenames
  rawCityFilename = "impact_city_raw.png"
  rawImpactFilename = "impact_raw.png"
  cityFilename = "impact_city_color.png"
  impactFilename = "impact_color.png"

  # Include night layer image to asses population density
  directory = os.path.dirname(__file__)
  nightFile = os.path.join(directory, "earth_night.png")
  nightImage = Image.open(nightFile)
  nightPixels = nightImage.load()

  # Size of the points
  brushSize = 20

  # Time range
  # 2023 CX1_t323_50/propagated_variants/: start "2023-Feb-12 22:00:00.000", end "2023-Feb-14"
  # 2004 MN4/openspace_variants/: start "2029-02-01", end "2029-08-17"
  # 2012 DA14/2012_03_05T06_24_12/openspace_variants/: start "2012-03-06", end "2013-09-30"
  # observers_test/: start "2023-02-25", end "2023-03-26"
  timeStart = spice.str2et("2012-03-06")
  timeEnd = spice.str2et("2013-09-30")

  # Get the raw map
  raw_impact_map(
    nVariants,
    kernel_list,
    timeStart,
    timeEnd,
    imageWidth,
    imageHeight,
    nightPixels,
    brushSize,
    rawCityFilename,
    rawImpactFilename
  )

  # Load the images that were just created again
  cityFile = os.path.join(directory, rawCityFilename)
  cityImage = Image.open(cityFile)
  cityImagePixels = cityImage.load()
  impactFile = os.path.join(directory, rawImpactFilename)
  impactImage = Image.open(impactFile)
  impactImagePixels = impactImage.load()

  # Transfer funciton settings
  maxValue = 400
  colorMap = cm.viridis

  # Apply a tranfer function to the map to get color
  color_impact_map(
    cityImage,
    impactImage,
    cityImagePixels,
    impactImagePixels,
    imageWidth,
    imageHeight,
    cityFilename,
    impactFilename,
    maxValue,
    colorMap
  )

  # Reset and unload all kernels
  kernelCounter = 0
  for kernel in kernel_list:
    progressBar(kernelCounter + 1, nVariants, "Unloading kernels")

    # Make sure it is a kernel file and not a directory
    kernelFile = os.path.join(input_path, kernel)
    if not os.path.isfile(kernelFile):
      continue

    # Unload the kernel file for this variant
    spice.unload(kernelFile)
    kernelCounter += 1

  print("--- %s seconds ---" % (time.time() - start_time))


# Impactors 2004 MN4: 88, 218, 372, 426, 498, 733, 738, 881, 893, 949, 993

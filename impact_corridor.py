import numpy as np
import os
import math
from PIL import Image, ImageFilter, ImageDraw

# Initialize SPICE
import spiceypy as spice
METAKERNEL = 'meta-kernel.tm'
spice.furnsh(METAKERNEL)

# List of valid Relational operators:
# "=", "<", ">", "LOCMIN", "ABSMIN", "ABSMIN", "LOCMAX", "ABSMAX", "ABSMAX"

# Main
if __name__ == "__main__":
  # Offset spice Id to not collide with any existing NAIF id
  id_offset = 1000000

  # Get the kernel files
  input_path = "./data/2012 DA14/2012_03_05T06_24_12/openspace_variants/"
  kernel_list = os.listdir(input_path)
  kernel_list.sort()

  # Set up the image
  imageWidth = 5400
  imageHeight = 2700
  image = Image.new("RGBA", (imageWidth, imageHeight), (0, 0, 0, 0))

  # Include night layer image to asses population density
  dirName = os.path.dirname(__file__)
  nightFile = os.path.join(dirName, "earth_night.png")
  nightImage = Image.open(nightFile)
  nightPixels = nightImage.load()

  # Convert to an image we can draw on
  drawImage = ImageDraw.Draw(image)
  brushSize = 20

  # Set the reference distance to Earth
  earth_radius = 6357 # minimum radius in Km

  # Specify the time range to search in
  # 2023 CX1_t323_50/propagated_variants/: start "2023-Feb-12 22:00:00.000", end "2023-Feb-14"
  # 2004 MN4/openspace_variants/: start "2029-02-01", end "2029-08-17"
  # 2012 DA14/2012_03_05T06_24_12/openspace_variants/: start "2012-03-06", end "2013-09-30"
  timeStart = spice.str2et("2012-03-06")
  timeEnd = spice.str2et("2013-09-30")
  timerange = spice.cell_double(200)
  spice.wninsd(timeStart, timeEnd, timerange)

  # Loop over the variants
  variant_id = id_offset
  for variant in kernel_list:
    variant_name = str(variant_id)

    # Make sure it is a kernel file and not a directory
    variant_kernel = os.path.join(input_path, variant)
    if not os.path.isfile(variant_kernel):
      print(variant_kernel, "is not a file")
      variant_id += 1
      continue

    # Load the kernel file for this variant
    spice.furnsh(variant_kernel)

    # Get times when the variant is close enough to Earth
    result = spice.gfdist(
      variant_name,       # Name of the target body
      "NONE",             # Aberration correction flag
      "EARTH",            # Name of the observing body
      "<",                # Relational operator (example <, = or >)
      earth_radius,       # Reference value
      0.0,                # Adjustment value for absolute extrema searches
      spice.spd(),        # Step size used for locating extrema and roots (The number of seconds in a day)
      100,                # Workspace window interval count (from example)
      timerange           # SPICE window to which the search is confined (from example)
    )

    # Check that we got a valid result
    if spice.wncard(result) == 0:
      #print("Variant", variant_name, "does not impact Earth")
      spice.unload(variant_kernel)
      variant_id += 1
      continue

    # Get results
    for i in range(spice.wncard(result)):
      # Get time range when object is within the reference distance
      [start, stop] = spice.wnfetd(result, i)
      impactTime = spice.timout(start, "YYYY MON DD HR:MN:SC.###", 41)

      # Get cartesian X, Y, Z position of object at the start time
      [position, lt]= spice.spkpos(
        variant_name,     # Target body name to check position for
        start,            # Time for the position
        "J2000",          # Reference frame of output position vector
        "NONE",           # Aberration correction flag
        "EARTH"           # Observing body name, reference frame of output position
      )

      # Get the distance, should be close to the reference distance set
      distance = spice.vnorm(position)

      print("Variant", variant_name, "impacts Earth on", impactTime, "distance:", distance)
      #print("position", position)

      # Get lat long coordinate
      [alt, long, lat] = spice.reclat(position)
      lat = math.degrees(lat)
      long = math.degrees(long)
      long += 180 # The international day time line is the 0 line (don't know why)
      print("position", lat, long, alt)

      # Normalize coordinate to be within 0 to 1 range
      lat *= -1 # Flip the y axis for the image
      pixelLat = (lat + 90)/180
      pixelLong = (long + 180)/360

      # Find corresponding pixel in equirectangular texture
      pixelY = round(pixelLat * imageHeight)
      pixelX = round(pixelLong * imageWidth)
      print("pixel", pixelX, pixelY)

      # Get color on night image, sample an area around the pixel and add up the color
      nightColorRed = 0
      for j in range(-int(brushSize/2), int(brushSize/2) + 1):
        for k in range(-int(brushSize/2), int(brushSize/2) + 1):
          pixedlcoord = pixelX + j, pixelY + k
          if (pixedlcoord[0] < 0 or pixedlcoord[1] < 0 or pixedlcoord[0] >= imageWidth or pixedlcoord[1] >= imageHeight):
            # Dont cross the borders of the image
            continue

          # Only use the red channel in the night layer
          nightColorRed += nightPixels[pixedlcoord][0]

      # Normaize the color
      nightColorRed = np.clip(round(nightColorRed / (brushSize * brushSize)), 0, 255)
      #print("resulting color", nightColorRed)

      # Draw circle
      shape = [(int(pixelX) - brushSize, pixelY - brushSize), (pixelX + brushSize, pixelY + brushSize)]
      drawImage.ellipse(shape, fill=(nightColorRed, nightColorRed, nightColorRed))

    # Reset
    spice.unload(variant_kernel)
    variant_id += 1

  # Make the whole image blurry
  blurryImage = drawImage._image.filter(ImageFilter.GaussianBlur(radius = int(brushSize/4)))
  #blurryImage.show()

  # Save the resulting image
  blurryImage.save("impact.png")

# Impactors 2004 MN4: 88, 218, 372, 426, 498, 733, 738, 881, 893, 949, 993
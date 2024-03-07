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
  input_path = "./data/2023 CX1_t323_50/propagated_variants/"
  kernel_list = os.listdir(input_path)
  kernel_list.sort()

  # Set up the image
  imageWidth = 5400
  imageHeight = 2700
  image = Image.new("RGB", (imageWidth, imageHeight), (0, 0, 0))

  # Convert to an image we can draw on
  drawImage = ImageDraw.Draw(image)
  brushSize = 20

  # Set the reference distance to Earth
  earth_radius = 6357 # minimum radius in Km

  # Specify the time range to search in
  timeStart = spice.str2et ("2023-Feb-12 22:00:00.000")
  timeEnd = spice.str2et ("2023-Feb-14")
  timerange = spice.cell_double(200)
  spice.wninsd(timeStart, timeEnd, timerange)

  # Loop over the variants
  variant_id = id_offset
  for variant in kernel_list:
    variant_name = str(variant_id)

    # Make sure it is a kernel file and not a directory
    variant_kernel = os.path.join(input_path, variant)
    if  not os.path.isfile(variant_kernel):
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
      print("Variant", variant_name, "does not impact Earth")
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

      # Normalize coordinate to be withing 0 to 1 range
      lat *= -1 # Flip the y axis for the image
      pixelLat = (lat + 90)/180
      pixelLong = (long + 180)/360

      # Find corresponding pixel in equirectangular texture
      pixelY = pixelLat * imageHeight
      pixelX = pixelLong * imageWidth
      print("pixel", int(pixelX), int(pixelY))

      # Paint a splat at the coresponding spot
      # image.putpixel((int(pixelX), int(pixelY)), (255, 255, 255))
      shape = [(int(pixelX) - brushSize, int(pixelY) - brushSize), (int(pixelX) + brushSize, int(pixelY) + brushSize)]
      drawImage.ellipse(shape, fill=(255, 255, 255))

    # Reset
    spice.unload(variant_kernel)
    variant_id += 1

  # Make the dots blurry and save the resulting image
  blurryImage = drawImage._image.filter(ImageFilter.GaussianBlur(radius = int(brushSize/2)))
  #blurryImage.show()
  blurryImage.save("impact.png")

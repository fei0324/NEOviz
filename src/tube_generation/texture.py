import os
import numpy as np
import matplotlib.pyplot as plt

# Define version numbers for the custom texture file format
# This is a uint8 value (max 255)
major_version = 0
minor_version = 2

EPSILON = 1e-4


def calculateTextureCoordinates(samples_2d, range_x=None, range_y=None):
    """Calculate UV coordinates and bounds for planar samples.

    ``samples_2d`` uses the polygon pipeline's canonical ``(N, 2)`` layout.
    The horizontal coordinate is inverted to match the convention used by the
    existing ellipse tube textures.
    """

    samples = np.asarray(samples_2d, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples_2d must have shape (N, 2)")
    if len(samples) < 3 or not np.all(np.isfinite(samples)):
        raise ValueError("samples_2d must contain at least three finite points")

    if range_x is None:
        range_x = np.array([np.min(samples[:, 0]), np.max(samples[:, 0])])
    else:
        range_x = np.asarray(range_x, dtype=float)
    if range_y is None:
        range_y = np.array([np.min(samples[:, 1]), np.max(samples[:, 1])])
    else:
        range_y = np.asarray(range_y, dtype=float)
    if range_x.shape != (2,) or range_y.shape != (2,):
        raise ValueError("range_x and range_y must each contain two values")
    if not np.all(np.isfinite(range_x)) or not np.all(np.isfinite(range_y)):
        raise ValueError("texture bounds must contain finite values")
    width = range_x[1] - range_x[0]
    height = range_y[1] - range_y[0]
    if width <= 0.0 or height <= 0.0:
        raise ValueError("texture bounds must have positive width and height")

    u = 1.0 - (samples[:, 0] - range_x[0]) / width
    v = (samples[:, 1] - range_y[0]) / height
    return np.column_stack((u, v)), range_x, range_y

def writeTexture(directory, filename, image_matrix, num_channels, resolution, min_values, 
                 max_values):
    """
    Write the given texture to a custom binary file format (.osimg)
    
    Input: 
        directory: The file directory where the texture file should be saved on disk
        filename: The name of the texture file to be used (without extension)
        image_matrix: A 2D numpy array containing the image data. Every pixel contains
                      num_channels of data
        num_channels: The number of channels in the given image data
        resolution: The resolution of the image, assumed to be square
        min_values: The minimum values in the image data for each available channel
        max_values: The maximum values in the image data 
    Output:
        full_filename: The full filename of the saved texture file
    """

    # Create the file, use 5 leading 0s for the timesteps
    full_filename = f"{filename:05d}.osimg"
    file_path = os.path.join(directory, full_filename)

    # Open the file for binary writing
    file = open(file_path, "wb")

    # Write the header with some meta data
    # First write the version number
    file.write(np.array([major_version, minor_version], dtype = np.uint8).tobytes())

    # Then the image size
    file.write(np.array([resolution, resolution], dtype = np.uint32).tobytes())

    # Write how many channels the image have
    
    file.write(np.uint8(num_channels).tobytes())

    # Then the min values and max values
    file.write(min_values.astype(np.float32).tobytes())
    file.write(max_values.astype(np.float32).tobytes())

    # Write the image data as 32-bit floats
    file.write(image_matrix.astype(np.float32).tobytes())

    # Close the file
    file.close()

    return full_filename


def generateDensityTexture(range_x, range_y, resolution, variants_2D):
    """
    Generate a density texture from the 2D variant positions using a gaussian brush.

    Input:
        range_x: The min and max values defining the range of x values for the texture.
                 I.e. the min and max values of the ellipse samples points in x
        range_y: The min and max values defining the range of y values for the texture.
        resolution: The resolution of the texture to generate, assumed to be square
        variants_2D: The intersected variant positions inside the ellipse in 2D
    Output:
        image_matrix: The generated density image matrix
        min_value: The minimum density value in the texture
        max_value: The maximum density value
    """

    # Settings. TODO: Make this configurable on function call
    brush_size = 5 # This is the radius of the brush in pixels, minus the center point
    sigma = int(brush_size / 2.0) # Standard deviation for the Gaussian brush
    kernel_size = brush_size * 2 + 1

    # Generate a Gaussian kernel for the brush
    xx = np.linspace(-brush_size, brush_size, kernel_size)
    yy = np.linspace(-brush_size, brush_size, kernel_size)
    xv, yv = np.meshgrid(xx, yy)
    
    # Create a normalized Gaussian kernel
    # The (xv**2 + yv**2) part is simplified from the original formula since we know that
    # the kernel is centered around the origin. Therfore the distances becomes the length
    # of the vector from the origin). Original formula from: 
    # https://www.geeksforgeeks.org/machine-learning/gaussian-kernel/
    gaussian_kernel = np.exp(-(xv**2 + yv**2) / (2 * sigma**2))
    assert gaussian_kernel.shape == (kernel_size, kernel_size), \
        "incorrect shape of the gaussian kernel"

    # Create a blank square texture with the given resolution
    image_matrix = np.zeros((resolution, resolution))

    # Plot the variants onto the texture
    for v in range(variants_2D.shape[1]):
        # Clamp the points to fit within the texture given the ellipse samples range in
        # x and y
        variant = variants_2D[:, v]
        x_index = int(
            (variant[0] - range_x[0]) / (range_x[1] - range_x[0]) * (resolution - 1)
        )
        y_index = int(
            (variant[1] - range_y[0]) / (range_y[1] - range_y[0]) * (resolution - 1)
        )

        # Make sure the indices are within bounds
        if x_index <= 0 or resolution <= x_index or y_index <= 0 or resolution <= y_index:
            print("Variant index out of bounds: (", x_index, ",", y_index, ")")
            print("This variant will be clamped to fit within the texture")
            x_index = max(0, min(resolution - 1, x_index))
            y_index = max(0, min(resolution - 1, y_index))

        # Each variant should be "painted" onto the texture using a gaussian brush that
        # superimposes to any previously "painted" variants. This gives a density effect.
        # Determin the area of the image matrix that should be affected by the guassian
        # brush centered at (x_index, y_index)
        x_start = max(0, x_index - brush_size)
        x_end = min(resolution, x_index + brush_size + 1)

        y_start = max(0, y_index - brush_size)
        y_end = min(resolution, y_index + brush_size + 1)

        # Determine the area of the gaussian kernel that should be used, normally all of
        # it would be used but there might be out of bounds issues near the edges
        kernel_x_start = max(0, brush_size - x_index)
        kernel_x_end = min(kernel_size, resolution - x_index + brush_size)

        kernel_y_start = max(0, brush_size - y_index)
        kernel_y_end = min(kernel_size, resolution - y_index + brush_size)

        # Make sure the size for the image and the kernel are the same size
        if image_matrix[x_start:x_end, y_start:y_end].shape != gaussian_kernel[
                kernel_x_start:kernel_x_end, kernel_y_start:kernel_y_end
            ].shape:
            print("\033[41mError\033[0m")
            print("Variant image index", x_index, y_index)
            print("image shape", image_matrix[x_start:x_end, y_start:y_end].shape)
            print("kernel shape", gaussian_kernel[kernel_x_start:kernel_x_end,
                                                  kernel_y_start:kernel_y_end].shape)
            assert False, "Mismatched shapes when applying gaussian kernel to image"

        # Apply the gaussian kernel to the image matrix. NOTE: The end ranges are one past
        # the last index, excluded
        image_matrix[x_start:x_end, y_start:y_end] += gaussian_kernel[
            kernel_x_start:kernel_x_end, kernel_y_start:kernel_y_end
        ]

    # Transpose the image matrix to get the correct orientation in OpenSpace
    image_matrix = image_matrix.T

    # Find the largest and smallest density value in the texture
    min_value = np.min(image_matrix)
    max_value = np.max(image_matrix)

    # Return the image matrix with its min and max values
    return image_matrix, min_value, max_value


def generatePointsTexture(range_x, range_y, resolution, variants_2D):
    """
    Generate a points position texture from the 2D variant positions using a hard square
    brush.

    Input:
        range_x: The min and max values defining the range of x values for the texture.
                 I.e. the min and max values of the ellipse samples points in x
        range_y: The min and max values defining the range of y values for the texture.
        resolution: The resolution of the texture to generate, assumed to be square
        variants_2D: The intersected variant positions inside the ellipse in 2D
    Output:
        image_matrix: The generated points positions image matrix
        min_value: The minimum value in the texture. In this case, known to be 0
        max_value: The maximum value. In this case we know it to be 1
    """

    # Settings. TODO: Make this configurable on function call
    brush_size = 2 # This is the radius of the brush in pixels, minus the center point

    # Create a blank square texture with the given resolution
    image_matrix = np.zeros((resolution, resolution))

    # Plot the variants onto the texture
    for v in range(variants_2D.shape[1]):
        # Clamp the points to fit within the texture given the ellipse samples range in
        # x and y
        variant = variants_2D[:, v]
        x_index = int(
            (variant[0] - range_x[0]) / (range_x[1] - range_x[0]) * (resolution)
        )
        y_index = int(
            (variant[1] - range_y[0]) / (range_y[1] - range_y[0]) * (resolution)
        )
        x_index = max(0, min(resolution - 1, x_index))
        y_index = max(0, min(resolution - 1, y_index))

        # Each variant should be "painted" onto the texture using a hard square brush that
        # overwrites any previously "painted" variants.
        # Determin the area of the image matrix that should be affected by the brush
        # centered at (x_index, y_index)
        x_start = max(0, x_index - brush_size)
        x_end = min(resolution, x_index + brush_size + 1)

        y_start = max(0, y_index - brush_size)
        y_end = min(resolution, y_index + brush_size + 1)

        # Apply the hard square brush to the image matrix. NOTE: The end ranges are one
        # past the last index, excluded
        image_matrix[x_start:x_end, y_start:y_end] = 1.0

    # Transpose the image matrix to get the correct orientation in OpenSpace
    image_matrix = image_matrix.T

    # Return the image matrix with its min and max values
    return image_matrix, 0.0, 1.0


def generateTimeLagsTexture(range_x, range_y, resolution, variants_2D, time_lags):
    """
    Generate a time lag texture from the given time lags using a hard square brush.

    Input:
        range_x: The min and max values defining the range of x values for the texture.
                 I.e. the min and max values of the ellipse samples points in x
        range_y: The min and max values defining the range of y values for the texture.
        resolution: The resolution of the texture to generate, assumed to be square
        variants_2D: The intersected variant positions inside the ellipse in 2D
        time_lags: The abstract time lag value for each variant in variants_2D, same order
    Output:
        image_matrix: The generated points positions image matrix
        min_value: The minimum value in the texture. In this case, known to be 0
        max_value: The maximum value. In this case we know it to be 1
    """

    # Settings. TODO: Make this configurable on function call
    brush_size = 2 # This is the radius of the brush in pixels, minus the center point

    # Find the time lag range to make sure that all values we write are normalized btween
    # 0 and 1. And the background is -0.1 to not interfer with the time lag values
    min_time_value = np.min(time_lags)
    max_time_value = np.max(time_lags)

    # Create a blank square texture with the given resolution
    time_matrix = np.full((resolution, resolution), -0.1)
    min_range_matrix = np.full((resolution, resolution), np.inf)
    max_range_matrix = np.full((resolution, resolution), -np.inf)
    range_matrix = np.zeros((resolution, resolution))

    # Plot the variants onto the texture
    for v in range(variants_2D.shape[1]):
        # Clamp the points to fit within the texture given the ellipse samples range in
        # x and y
        variant = variants_2D[:, v]
        x_index = int(
            (variant[0] - range_x[0]) / (range_x[1] - range_x[0]) * (resolution)
        )
        y_index = int(
            (variant[1] - range_y[0]) / (range_y[1] - range_y[0]) * (resolution)
        )
        x_index = max(0, min(resolution - 1, x_index))
        y_index = max(0, min(resolution - 1, y_index))

        # Each variant should be "painted" onto the texture using a hard square brush that
        # overwrites any previously "painted" variants.
        # Determin the area of the image matrix that should be affected by the brush
        # centered at (x_index, y_index)
        x_start = max(0, x_index - brush_size)
        x_end = min(resolution, x_index + brush_size + 1)

        y_start = max(0, y_index - brush_size)
        y_end = min(resolution, y_index + brush_size + 1)

        # Check if there will be any overlap with previously painted variants. This need
        # to happen before we paint with this variants value
        value = time_lags[v]
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                if time_matrix[x, y] > -0.1:
                    # Overlap, then check if the time lag between the two samples are
                    # different
                    # If the new value is smller than the old, update the min
                    if value < time_matrix[x, y]:
                        min_range_matrix[x, y] = value

                        # Check the cooresponding max value. If nothing has been written
                        # already, then put the old value there to indicate that there
                        # has been overlap here and by how much the time lag has deviated
                        # between the two overlapping samples. However if multiple
                        # overlaps have happened, we only want to keep the largest max
                        # value
                        max_range_matrix[x, y] = max(
                            max_range_matrix[x, y],
                            time_matrix[x, y]
                        )

                    # Or if it is larger than the old, update the max
                    if value > time_matrix[x, y]:
                        max_range_matrix[x, y] = value

                        # Check the cooresponding min value. If nothing has been written
                        # already, then put the old value there to indicate that there
                        # has been overlap here and by how much the time lag has deviated
                        # between the two overlapping samples. However if multiple
                        # overlaps have happened, we only want to keep the smallest min
                        # value
                        min_range_matrix[x, y] = min(
                            min_range_matrix[x, y],
                            time_matrix[x, y]
                        )

        # Apply the hard square brush to the image matrix. NOTE: The end ranges are one
        # past the last index, excluded.
        time_matrix[x_start:x_end, y_start:y_end] = \
            (value - min_time_value) / (max_time_value - min_time_value)

    # Get the range matrix by subtracting the min from the max
    for x in range(resolution):
        for y in range(resolution):
            # Make sure that any pixels that are untouched by the variants are set to -0.1
            if min_range_matrix[x, y] == np.inf or max_range_matrix[x, y] == -np.inf:
                range_matrix[x, y] = -0.1
            else:
                # This makes sure that we always have a positive range value, and therefor
                # will not interfer with the untouched background
                range_matrix[x, y] = abs(max_range_matrix[x, y] - min_range_matrix[x, y])

    # Transpose the image matrix to get the correct orientation in OpenSpace
    time_matrix = time_matrix.T
    range_matrix = range_matrix.T

    # Find the largest and smallest density value in the texture
    min_value = np.min(time_matrix)
    max_value = np.max(time_matrix)
    max_range = np.max(range_matrix)

    # Return the image matrix with its min and max values
    return time_matrix, min_value, max_value, range_matrix, -0.1, max_range


def combineImages(images_list, resolution):
    """
    """

    # Determine the number of channels from the length of the provided images list
    num_channels = len(images_list)

    # Create a blank image matrix with the given resolution and number of channels
    combined_image_matrix = np.zeros((resolution, resolution, num_channels))

    # Fill the image matrix with the provided images. Where each image is one channel, use
    # the order they were provided in the list
    for i in range(num_channels):
        combined_image_matrix[:, :, i] = images_list[i]
        
    # Return the combined image matrix
    return combined_image_matrix    


def generateTexture(samples_range_x, samples_range_y, resolution, directory, time_step, 
                    texture_coordinates, variants_2D, time_lags):
    """
    """

    # Generate the density texture
    density_matrix, min_density, max_density = generateDensityTexture(
        samples_range_x,
        samples_range_y,
        resolution,
        variants_2D
    )

    # Generate the sample position texture
    points_matrix, min_points, max_points = generatePointsTexture(
        samples_range_x,
        samples_range_y,
        resolution,
        variants_2D
    )

    # Generate the time lag texture
    time_matrix, min_lag, max_lag, range_matrix, min_range, max_range = \
    generateTimeLagsTexture(
        samples_range_x,
        samples_range_y,
        resolution,
        variants_2D,
        time_lags
    )

    # Combine all textures into one multi-channel texture
    images_list = [density_matrix, points_matrix, time_matrix, range_matrix]
    combined_image_matrix = combineImages(images_list, resolution)

    # Combine min and max values for all images/channels
    min_values = np.array([min_density, min_points, min_lag, min_range])
    max_values = np.array([max_density, max_points, max_lag, max_range])

    # Write the texture to file
    full_filename = writeTexture(
        directory,
        time_step,
        combined_image_matrix,
        len(images_list),
        resolution,
        min_values, 
        max_values
    )

    # Return the full filename of the written texture
    return full_filename

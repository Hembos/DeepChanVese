import numpy as np
from math import ceil
from numba import jit

def apply_kuwahara_filter(image: np.ndarray, filter_size: int = 3) -> np.ndarray:
    """Apply kuwahara filter for image. The Kuwahara filter can be defined as a 
    non-linear smoothing filter that does not compromise the sharpness of the 
    image or the positions of the edges

    Args:
        image (np.ndarray): image pixel array with (W x H) size
        filter_size (int, optional): kuwahara window size. Must be odd. Defaults to 3.

    Returns:
        np.ndarray: smoothed image
    """
    if filter_size % 2 != 1:
        raise ValueError("Filter size must be odd")
    
    res_image = np.zeros(image.shape)

    height, width = image.shape

    quadrant_size = ceil(filter_size / 2)

    for y in range(height):
        for x in range(width):
            tl_x = x - quadrant_size
            tl_y = y - quadrant_size

            window_indices = np.indices((filter_size, filter_size))
            window_indices[0] += tl_y
            window_indices[1] += tl_x
            window_indices[0][window_indices[0] < 0] = 0
            window_indices[1][window_indices[1] < 0] = 0

            q1 = window_indices[:,:quadrant_size,:quadrant_size].reshape(2, quadrant_size**2)
            q2 = window_indices[:,quadrant_size - 1:,:quadrant_size].reshape(2, quadrant_size**2)
            q3 = window_indices[:,:quadrant_size,quadrant_size - 1:].reshape(2, quadrant_size**2)
            q4 = window_indices[:,quadrant_size - 1:,quadrant_size - 1:].reshape(2, quadrant_size**2)

            qudrants = [image[q1[0], q1[1]],
                        image[q2[0], q2[1]],
                        image[q3[0], q3[1]],
                        image[q4[0], q4[1]]]
            
            div = [np.std(quadrant) for quadrant in qudrants]

            qudrant = qudrants[np.argmin(div)]
            res_image[y][x] = np.mean(qudrant)

    return res_image




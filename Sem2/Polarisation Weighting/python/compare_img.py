import numpy as np
import matplotlib.pyplot as plt
import cv2

def difference_heatmap(image1, image2, cmap='jet'):
    """
    Compute the difference between two images and produce a normalized heatmap of the difference.

    Parameters:
    - image1: First image (numpy array).
    - image2: Second image (numpy array).
    - cmap: Colormap for the heatmap (default is 'jet').

    Returns:
    - norm_diff: The normalized difference matrix.
    - heatmap: The normalized heatmap of the difference.
    """

    # Ensure images are the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must be of the same size.")

    # Convert images to grayscale if they are in RGB
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the images
    diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))

    # Normalize the difference to the range [0, 1]
    norm_diff = cv2.normalize(diff, None, 0, 1, cv2.NORM_MINMAX)

    # Generate the heatmap
    heatmap = plt.get_cmap(cmap)(norm_diff)

    # Convert to an RGB image (remove the alpha channel if present)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)

    return norm_diff, heatmap

# Example usage:
image1 = cv2.imread('python/test_im/ball/ball_0.png')
image2 = cv2.imread('python/test_im/ball/ball_90.png')

norm_diff, heatmap = difference_heatmap(image1, image2)

# Display the heatmap using matplotlib with a colorbar
plt.imshow(norm_diff, cmap='jet')
plt.colorbar(label='Difference Intensity')
plt.title('Heatmap of Image Differences')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import WeightingEstimate
import ShiftEstimate
import Autocorrelation
import skimage.restoration as sk
from scipy.interpolate import RegularGridInterpolator

def seperate_imgs_into_patches(img, patch_size):
    """
    Separate an image into patches with dynamically calculated overlap, ensuring
    all patches fit into the image evenly without leaving any areas uncovered.
    
    img: the image to separate
    patch_size: the size of each patch
    
    Returns:
    - patches: list of image patches
    - patch_info: list of tuples containing (x0, y0, x1, y1) coordinates for each patch
    - overlap: the automatically calculated overlap
    """
    img_height, img_width = img.shape[:2]
    
    # Calculate the number of patches along the width and height
    num_patches_x = int(np.ceil(img_width / patch_size))
    num_patches_y = int(np.ceil(img_height / patch_size))
    
    # Calculate the overlap needed to ensure patches fit evenly across the image
    step_x = img_width / num_patches_x
    step_y = img_height / num_patches_y
    
    overlap_x = patch_size - step_x
    overlap_y = patch_size - step_y
    
    patches = []
    patch_info = []
    
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            x0 = int(i * step_x)
            y0 = int(j * step_y)
            x1 = min(x0 + patch_size, img_width)
            y1 = min(y0 + patch_size, img_height)
            
            # Extract the patch
            patch = img[y0:y1, x0:x1]
            patches.append(patch)
            patch_info.append((x0, y0, x1, y1))
    
    return patches, patch_info, (overlap_x, overlap_y)



def combine_patches_into_image_with_overlap(patches, patch_info, original_shape):
    """
    Combine the patches back into the original image, accounting for overlap using weighted blending.
    
    patches: list of deconvolved image patches
    patch_info: list of tuples containing (x0, y0, x1, y1) coordinates for each patch
    original_shape: shape of the original image (height, width, channels)
    
    Returns the re-constructed image.
    """
    # Create a blank image and a weight map to handle overlaps
    reconstructed_image = np.zeros(original_shape)
    weight_map = np.zeros(original_shape[:2])
    
    # Reconstruct the image by placing each patch at its original location
    for patch, (x0, y0, x1, y1) in zip(patches, patch_info):
        patch_height, patch_width = patch.shape[:2]
        # Create a weight matrix for blending. Use a linear gradient (e.g., from 0 to 1) across the patch.
        weight_x = np.linspace(0, 1, patch_width).reshape(1, -1)
        weight_y = np.linspace(0, 1, patch_height).reshape(-1, 1)
        
        # Compute the final weight as the outer product of the two gradients
        weight = np.minimum(weight_x, weight_y)  # More complex blending can use np.outer(weight_x, weight_y)
        # rgb
        weight_rgb = np.dstack([weight] * 3)
        # Apply the patch and its weight to the respective region in the image
        reconstructed_image[y0:y1, x0:x1] += patch * weight_rgb
        weight_map[y0:y1, x0:x1] += weight
    
    # Avoid division by zero in areas without patches
    weight_map[weight_map == 0] = 1
    
    # Normalize the reconstructed image by dividing by the weight map
    # reconstructed_image /= weight_map
    reconstructed_image /= np.dstack([weight_map] * 3)
    
    return reconstructed_image
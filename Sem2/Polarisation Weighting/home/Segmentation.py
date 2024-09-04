import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb

def segment_image_patches(img, shift_val = 6):
    # get image size
    height, width = img.shape[:2]
    divisor = (6 * shift_val)
    get_max_patches_x = (width // divisor)
    get_max_patches_y = (height // divisor)

    # Create a copy of the image for visualization (drawing rectangles)
    img_with_rectangles = img.copy()

    patches = []
    
    # Iterate over patches in y (height) and x (width)
    for y in range(get_max_patches_y + 1):
        for x in range(get_max_patches_x + 1):
            # Determine the x and y start coordinates of the patch
            x_start = x * divisor
            y_start = y * divisor
            
            # Determine the x and y end coordinates (handle edge cases)
            x_end = min((x + 1) * divisor, width)
            y_end = min((y + 1) * divisor, height)
            
            patch = img[y_start:y_end, x_start:x_end]
            patches.append(patch)

    return patches

def recombine_patches(patches, original_height, original_width, shift_val=6):
    # Create a blank canvas for the recombined image
    divisor = (6 * shift_val)
    recombined_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # Calculate the number of patches in both dimensions
    get_max_patches_x = (original_width // divisor)
    get_max_patches_y = (original_height // divisor)

    # Iterate through the patches and place them in the correct position
    patch_index = 0
    for y in range(get_max_patches_y + 1):
        for x in range(get_max_patches_x + 1):
            # Determine the x and y start coordinates for placement
            x_start = x * divisor
            y_start = y * divisor
            
            # Get the size of the current patch
            patch = patches[patch_index]
            patch_height, patch_width = patch.shape[:2]
            
            # Place the patch back into the recombined image
            recombined_image[y_start:y_start + patch_height, x_start:x_start + patch_width] = patch
            
            # Increment the patch index
            patch_index += 1

    return recombined_image

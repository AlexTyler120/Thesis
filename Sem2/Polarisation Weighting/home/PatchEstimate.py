import matplotlib.pyplot as plt
import numpy as np
import WeightingEstimate
import ShiftEstimate
import Autocorrelation
import skimage.restoration as sk
from scipy.interpolate import RegularGridInterpolator
import cv2
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
    reconstructed_image = np.zeros(original_shape[:2])
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
        # weight_rgb = np.dstack([weight] * 3)
        # Apply the patch and its weight to the respective region in the image
        reconstructed_image[y0:y1, x0:x1] += patch * weight
        weight_map[y0:y1, x0:x1] += weight
    
    # Avoid division by zero in areas without patches
    weight_map[weight_map == 0] = 1
    
    # Normalize the reconstructed image by dividing by the weight map
    reconstructed_image /= weight_map
    # reconstructed_image /= np.dstack([weight_map] * 3)
    
    return reconstructed_image

def blend_patches(patches, patch_info, img_size, overlap):
    """
    Reconstruct the full image from patches using a linear blending technique.
    
    patches: list of image patches
    patch_info: list of tuples containing (x0, y0, x1, y1) coordinates for each patch
    img_size: size of the original image (height, width)
    overlap: the overlap (overlap_x, overlap_y)
    
    Returns:
    - full_img: the reconstructed full image with blended patches
    """
    img_height, img_width = img_size
    overlap_x, overlap_y = overlap
    
    # Create an empty image to reconstruct
    full_img = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Create a weight matrix to accumulate blending weights
    weight_matrix = np.zeros((img_height, img_width), dtype=np.float32)
    
    for patch, (x0, y0, x1, y1) in zip(patches, patch_info):
        patch_height, patch_width = patch.shape[:2]
        
        # Create blending weights for this patch
        weight_x = np.ones((patch_height, 1))
        weight_y = np.ones((1, patch_width))
        
        if overlap_x > 0:
            weight_x = np.linspace(1, 0, int(overlap_x), endpoint=False).reshape(-1, 1)
            weight_x = np.vstack((weight_x, np.ones(patch_height - int(overlap_x)).reshape(-1, 1)))

        if overlap_y > 0:
            weight_y = np.linspace(1, 0, int(overlap_y), endpoint=False).reshape(1, -1)
            weight_y = np.hstack((weight_y, np.ones(patch_width - int(overlap_y)).reshape(1, -1)))

        weight = weight_x * weight_y
        
        # Add patch to the full image with blending
        full_img[y0:y1, x0:x1] += patch * weight
        weight_matrix[y0:y1, x0:x1] += weight
    
    # Normalize by the accumulated weight to avoid intensity shifts
    full_img /= (weight_matrix)  # Avoid division by zero
    
    # Clip to valid image range
    full_img = np.clip(full_img, 0, 255).astype(np.uint8)
    
    return full_img

def gaussian_blend_patches(patches, patch_info, img_size, overlap):
    """
    Reconstruct the full image from patches using Gaussian blending technique.
    
    patches: list of image patches
    patch_info: list of tuples containing (x0, y0, x1, y1) coordinates for each patch
    img_size: size of the original image (height, width)
    overlap: the overlap (overlap_x, overlap_y)
    
    Returns:
    - full_img: the reconstructed full image with Gaussian blended patches
    """
    img_height, img_width = img_size
    overlap_x, overlap_y = overlap
    
    # Create an empty image to reconstruct
    full_img = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Create a weight matrix to accumulate blending weights
    weight_matrix = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Define Gaussian kernel for smoothing
    def gaussian_kernel(size, sigma=1):
        if size <= 1:
            print(f"No overlap for size {size}")
            return np.ones(1)  # If there's no overlap, return ones
        kernel_1d = np.linspace(-(size // 2), size // 2, size)
        gauss = np.exp(-0.5 * (kernel_1d / sigma) ** 2)
        gauss = gauss / gauss.sum()
        return gauss
    
    # Create Gaussian weights for overlap regions
    gauss_x = gaussian_kernel(int(overlap_x * 2)) if overlap_x > 0 else np.ones(1)
    gauss_y = gaussian_kernel(int(overlap_y * 2)) if overlap_y > 0 else np.ones(1)
    
    for patch, (x0, y0, x1, y1) in zip(patches, patch_info):
        patch_height, patch_width = patch.shape[:2]
        
        # Create blending weights for this patch
        weight_x = np.ones((patch_height, 1))
        weight_y = np.ones((1, patch_width))
        
        if overlap_x > 0:
            # Make sure the sizes match before stacking
            overlap_len_x = min(len(gauss_x[:int(overlap_x)]), patch_height)
            weight_x = np.vstack((gauss_x[:overlap_len_x].reshape(-1, 1), np.ones(patch_height - overlap_len_x).reshape(-1, 1)))

        if overlap_y > 0:
            # Make sure the sizes match before stacking
            overlap_len_y = min(len(gauss_y[:int(overlap_y)]), patch_width)
            weight_y = np.hstack((gauss_y[:overlap_len_y].reshape(1, -1), np.ones(patch_width - overlap_len_y).reshape(1, -1)))

        weight = weight_x * weight_y
        
        # Add patch to the full image with Gaussian blending
        full_img[y0:y1, x0:x1] += patch * weight
        weight_matrix[y0:y1, x0:x1] += weight
    
    # Normalize by the accumulated weight to avoid intensity shifts
    full_img /= (weight_matrix + 1e-6)  # Avoid division by zero
    
    # Clip to valid image range
    full_img = np.clip(full_img, 0, 255).astype(np.uint8)
    
    return full_img

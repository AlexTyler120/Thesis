import matplotlib.pyplot as plt
import numpy as np
import WeightingEstimate
import ShiftEstimate
import Autocorrelation
import skimage.restoration as sk
from scipy.interpolate import RegularGridInterpolator
import cv2
import scipy as sp

def extract_image_patches_no_overlap(image, patch_size):
    """
    Splits an image into patches of the specified size.
    
    Parameters:
    - image: Input image as a numpy array (e.g., loaded using cv2.imread).
    - patch_size: Tuple of (patch_height, patch_width).
    
    Returns:
    - patches: List of image patches, where each patch is a numpy array.
    """
    patches = []
    img_height, img_width = image.shape[:2]
    patch_height, patch_width = patch_size
    padding_size = 5 

    # Loop over the image to extract patches, ignoring smaller patches at edges
    for y in range(0, img_height, patch_height):
        for x in range(0, img_width, patch_width):
            # Ensure patch size is valid (no smaller patches at the edges)
            if (y + patch_height <= img_height) and (x + patch_width <= img_width):
                patch = image[y:y+patch_height, x:x+patch_width]
                # Pad the patch with reflection padding on both x and y axes
                padded_patch = cv2.copyMakeBorder(
                    patch, padding_size, padding_size, padding_size, padding_size, 
                    borderType=cv2.BORDER_REFLECT
                )
                patches.append(padded_patch)
    
    return patches


def reconstruct_image_from_patches_no_overlap(patches, image_size, patch_size):
    """
    Reconstructs an image from patches.
    
    Parameters:
    - patches: List of image patches, where each patch is a numpy array.
    - image_size: Tuple of (image_height, image_width) representing the original image size.
    - patch_size: Tuple of (patch_height, patch_width) representing the size of each patch.
    
    Returns:
    - reconstructed_image: The reconstructed image as a numpy array.
    """
    image_height, image_width = image_size
    patch_height, patch_width = patch_size
    padding_size = 5
    # Create an empty array for the reconstructed image
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)

    patch_idx = 0
    # Loop over the image to place patches in the correct position
    for y in range(0, image_height, patch_height):
        for x in range(0, image_width, patch_width):
            # Ensure we are not placing smaller patches (ignored at edges)
            if (y + patch_height <= image_height) and (x + patch_width <= image_width):
                # Remove the padding by cropping the padding from all sides
                patch = patches[patch_idx]
                cropped_patch = patch[padding_size:-padding_size, padding_size:-padding_size]
                # Place the cropped patch back into the image
                reconstructed_image[y:y+patch_height, x:x+patch_width] = cropped_patch
                patch_idx += 1
    
    return reconstructed_image

def reconstruct_image_from_patches_no_overlap_with_quiver(patches, image_size, patch_size, w12_vals, channel):
    """
    Reconstructs an image from patches and adds a single quiver arrow for each patch based on dx, dy values.
    
    Parameters:
    - patches: List of image patches, where each patch is a numpy array.
    - image_size: Tuple of (image_height, image_width) representing the original image size.
    - patch_size: Tuple of (patch_height, patch_width) representing the size of each patch.
    - w12_vals: List of tuples (U, V) representing the dx, dy values for quiver plots for each patch.
    
    Returns:
    - reconstructed_image: The reconstructed image as a numpy array.
    """
    image_height, image_width = image_size
    patch_height, patch_width = patch_size
    padding_size = 5
    
    # Create an empty array for the reconstructed image
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)

    patch_idx = 0
    X_quiver = []
    Y_quiver = []
    U_quiver = []
    V_quiver = []

    # Loop over the image to place patches in the correct position
    for y in range(0, image_height, patch_height):
        for x in range(0, image_width, patch_width):
            # Ensure we are not placing smaller patches (ignored at edges)
            if (y + patch_height <= image_height) and (x + patch_width <= image_width):
                # Remove the padding by cropping the padding from all sides
                patch = patches[patch_idx]
                cropped_patch = patch[padding_size:-padding_size, padding_size:-padding_size]
                
                # Place the cropped patch back into the image
                reconstructed_image[y:y+patch_height, x:x+patch_width] = cropped_patch
                
                # Get the quiver dx, dy values for this patch
                U, V = w12_vals[patch_idx]  # Assuming w12_vals contains (U, V) for each patch
                if U > V:
                    U = -U
                    V = -V

                # Calculate the center of the current patch
                center_x = x + patch_width // 2
                center_y = y + patch_height // 2

                # Store the center and the quiver values
                X_quiver.append(center_x)
                Y_quiver.append(center_y)
                U_quiver.append(U)
                V_quiver.append(V)

                patch_idx += 1

    # Plot the reconstructed image with one quiver arrow per patch
    if channel == 0:
        colour = 'red'
    elif channel == 1:
        colour = 'green'
    elif channel == 2:
        colour = 'blue'
    else:
        colour = 'black'    
    
    plt.figure(figsize=(10, 10))
    plt.imshow(reconstructed_image, cmap='gray')
    plt.quiver(X_quiver, Y_quiver, U_quiver, V_quiver, color=colour, angles='xy', scale_units='xy', scale=0.1)
    plt.title(f"Reconstructed Image with Single Quiver Arrow per Patch for channel {colour}")
    
    return reconstructed_image

def create_full_quiver(image, image_size, patch_size, w12_vals):
    image_height, image_width = image_size
    patch_height, patch_width = patch_size
    patch_idx = 0
    X_quiver = []
    Y_quiver = []
    U_quiver = []
    V_quiver = []
    for y in range(0, image_height, patch_height):
        for x in range(0, image_width, patch_width):
            if (y + patch_height <= image_height) and (x + patch_width <= image_width):
                # Get the quiver dx, dy values for this patch
                U, V = w12_vals[patch_idx]  # Assuming w12_vals contains (U, V) for each patch
                if U > V:
                    U = -U
                    # V = -V
                    
                # Calculate the center of the current patch
                center_x = x + patch_width // 2
                center_y = y + patch_height // 2

                # Store the center and the quiver values
                X_quiver.append(center_x)
                Y_quiver.append(center_y)
                U_quiver.append(U)
                V_quiver.append(V)

                patch_idx += 1
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.quiver(X_quiver, Y_quiver, U_quiver, V_quiver, color='black', angles='xy', scale_units='xy', scale=0.1)
    plt.title(f"Reconstructed Image with Single Quiver Arrow per Patch for rgb")
    
def reconstruct_image_from_patches_with_overlap(patches, image_size, patch_size, stride_y, stride_x):
    """
    Reconstructs an image from overlapping patches by blending the overlaps.
    
    Parameters:
    - patches: List of windowed, overlapping image patches, where each patch is a numpy array.
    - image_size: Tuple of (image_height, image_width) representing the original image size.
    - patch_size: Tuple of (patch_height, patch_width) representing the size of each patch.
    - stride_y: Integer representing the vertical stride (patch_height - overlap_size).
    - stride_x: Integer representing the horizontal stride (patch_width - overlap_size).
    
    Returns:
    - reconstructed_image: The reconstructed image as a numpy array.
    """
    image_height, image_width = image_size
    patch_height, patch_width = patch_size
    
    # Create empty arrays for the reconstructed image and a weight matrix
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)
    weight_matrix = np.zeros((image_height, image_width), dtype=np.float32)

    patch_idx = 0
    # Loop over the image to place patches in the correct position
    for y in range(0, image_height - patch_height + 1, stride_y):
        for x in range(0, image_width - patch_width + 1, stride_x):
            patch = patches[patch_idx]
            reconstructed_image[y:y+patch_height, x:x+patch_width] += patch
            weight_matrix[y:y+patch_height, x:x+patch_width] += 1  # Track the number of overlaps
            patch_idx += 1
    
    # Normalize the reconstructed image by dividing by the weight matrix to blend overlaps
    reconstructed_image /= np.maximum(weight_matrix, 1)  # Avoid division by zero
    
    return reconstructed_image


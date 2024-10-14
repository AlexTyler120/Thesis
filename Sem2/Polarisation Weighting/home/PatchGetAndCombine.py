import matplotlib.pyplot as plt
import numpy as np
import WeightingEstimate
import ShiftEstimate
import Autocorrelation
import skimage.restoration as sk
from scipy.interpolate import RegularGridInterpolator
import cv2
import scipy as sp
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter 
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, generic_filter, median_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.filters import rank
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.stats import zscore
import pickle

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
    padding_size = patch_width//2

    # Loop over the image to extract patches, ignoring smaller patches at edges
    for y in range(0, img_height, patch_height):
        for x in range(0, img_width, patch_width):
            # Ensure patch size is valid (no smaller patches at the edges)
            if (y + patch_height <= img_height) and (x + patch_width <= img_width):
                
                patch = image[y:y+patch_height, x:x+patch_width]
                    
                padded_patch = cv2.copyMakeBorder(
                    patch, padding_size, padding_size, padding_size, padding_size, 
                    borderType=cv2.BORDER_REFLECT
                    # borderType=cv2.BORDER_REPLICATE
                    # borderType=cv2.BORDER_CONSTANT
                )
                # save image
                
                patches.append(padded_patch)
                
                # patches.append(patch)
    
    return patches



def reconstruct_image_patch_intensity(patches, deconvolved_patches, image_size, patch_size, channel, w12vals):
    image_height, image_width = image_size
    patch_height, patch_width = patch_size
    padding_size = patch_width//2
    # Create an empty array for the reconstructed image
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)
    patch_idx = 0
    I_0_total = np.zeros_like(reconstructed_image)
    I_90_total = np.zeros_like(reconstructed_image)

    angle_map = np.zeros_like(reconstructed_image)
    num_rows = image_height // patch_height  # Number of patches in vertical direction
    num_cols = image_width // patch_width    # Number of patches in horizontal direction
    
    magnitude_spectrum = np.zeros((image_height, image_width))
    print(f"Num Rows: {num_rows} and Num Cols: {num_cols}")
    row = 0
    for y in range(0, image_height, patch_height):
        col = 0
        for x in range(0, image_width, patch_width):
            # Ensure we are not placing smaller patches (ignored at edges)
            if (y + patch_height <= image_height) and (x + patch_width <= image_width):
                # Remove the padding by cropping the padding from all sides
                patch = patches[patch_idx]
                patch = patch[:,:, channel]
                deconvolved_patch = deconvolved_patches[patch_idx]
                w1, w2 = w12vals[patch_idx]  
                
                cropped_deconvolved_patch = deconvolved_patch[padding_size:-padding_size, padding_size:-padding_size]
                
                # Place the cropped patch back into the image
                

                # angles for each pixel
                angles = np.zeros_like(cropped_deconvolved_patch)
                mag_pix = np.zeros_like(cropped_deconvolved_patch)
                I_0_patch = np.zeros_like(cropped_deconvolved_patch)
                I_90_patch = np.zeros_like(cropped_deconvolved_patch)

                for i in range(patch_height):
                    for j in range(patch_width):
                        angles[i,j] = np.degrees(np.arctan2(w1, w2))
                        mag_pix[i, j] = np.sqrt(w1**2 + w2**2)
                        I_0_patch[i,j] = (cropped_deconvolved_patch[i,j]) * w2
                        I_90_patch[i, j] = w1 * cropped_deconvolved_patch[i, j]

                        
                reconstructed_image[y:y+patch_height, x:x+patch_width] = cropped_deconvolved_patch
                I_0_total[y:y+patch_height, x:x+patch_width] = I_0_patch
                I_90_total[y:y+patch_height, x:x+patch_width] = I_90_patch
                angle_map[y:y+patch_height, x:x+patch_width] = angles
                magnitude_spectrum[y:y+patch_height, x:x+patch_width] = mag_pix
                patch_idx += 1
            col += 1
        row += 1
    
    patches_row_width = patch_width * num_cols
    patches_col_height = patch_height * num_rows
    print(f"Row Width: {patches_row_width} and Col Height: {patches_col_height}")
    # crop images to this
    I_0_total = I_0_total[:patches_col_height, :patches_row_width]
    I_90_total = I_90_total[:patches_col_height, :patches_row_width]
    angle_map = angle_map[:patches_col_height, :patches_row_width]
    magnitude_spectrum = magnitude_spectrum[:patches_col_height, :patches_row_width]
    reconstructed_image = reconstructed_image[:patches_col_height, :patches_row_width]
    
    
    plt.figure()
    plt.imshow(angle_map, cmap='jet')
    plt.title(f"Angle Map channel {channel}")
    plt.colorbar()
    plt.figure()
    plt.imshow(magnitude_spectrum, cmap='jet')
    plt.title(f"Magnitude Spectrum channel {channel}")
    plt.title("Magnitude Spectrum")
    plt.colorbar()
    
    plt.figure()
    plt.imshow(I_0_total, cmap='Blues')
    plt.title(f"I_0 channel {channel}")
    plt.colorbar()
    plt.figure()
    plt.imshow(I_90_total, cmap='Reds')
    plt.title(f"I_90 channel {channel}")
    plt.colorbar()
    plt.figure()
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f"Reconstructed Image channel {channel}")
    
    return reconstructed_image

def extract_image_patches_overlap(image, patch_size):
    """
    Splits an image into overlapping patches of the specified size, with 50% overlap.
    
    Parameters:
    - image: Input image as a numpy array.
    - patch_size: Tuple of (patch_height, patch_width).
    
    Returns:
    - patches: List of image patches with 50% overlap, where each patch is a numpy array.
    """
    patches = []
    img_height, img_width = image.shape[:2]
    patch_height, patch_width = patch_size
    step_y = patch_height // 2
    step_x = patch_width // 2
    padding_size = patch_width//2
    # convert bgr to rgb
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Loop over the image to extract patches with 50% overlap
    for y in range(0, img_height - patch_height + 1, step_y):
        for x in range(0, img_width - patch_width + 1, step_x):
            if (y + patch_height <= img_height) and (x + patch_width <= img_width):
                
                patch = image[y:y + patch_height, x:x + patch_width]
                padded_patch = cv2.copyMakeBorder(
                    patch, padding_size, padding_size, padding_size, padding_size, 
                    borderType=cv2.BORDER_REFLECT
                    # borderType=cv2.BORDER_REPLICATE
                    # borderType=cv2.BORDER_CONSTANT
                )
                patches.append(padded_patch)
    
    return patches

def reconstruct_image_patch_intensity_overlap(patches, deconvolved_patches, image_size, patch_size, channel, w12vals):
    """
    Reconstructs the image from overlapping patches by averaging the overlapping areas and blending w1, w2.
    
    Parameters:
    - patches: List of original image patches.
    - deconvolved_patches: List of deconvolved patches.
    - image_size: Tuple of (image_height, image_width).
    - patch_size: Tuple of (patch_height, patch_width).
    - channel: The specific color channel to process.
    - w12vals: List of tuples containing (w1, w2) values for each patch.
    
    Returns:
    - reconstructed_image: The reconstructed image after blending the patches.
    """
    image_height, image_width = image_size
    patch_height, patch_width = patch_size
    padding_size = patch_width // 2

    # Create empty arrays for reconstructed images, angle, magnitude, and weight maps
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)
    I_0_total = np.zeros_like(reconstructed_image)
    I_90_total = np.zeros_like(reconstructed_image)
    angle_map = np.zeros_like(reconstructed_image)
    magnitude_spectrum = np.zeros_like(reconstructed_image)
    w1_map = np.zeros_like(reconstructed_image)
    w2_map = np.zeros_like(reconstructed_image)
    w_diff_map = np.zeros_like(reconstructed_image)
    weight_map = np.zeros((image_height, image_width), dtype=np.float32)

    patch_idx = 0
    step_y = patch_height // 2  # 50% overlap in vertical direction
    step_x = patch_width // 2    # 50% overlap in horizontal direction

    # Loop over the patches and place them in their respective positions, averaging overlapping regions
    for y in range(0, image_height - patch_height + 1, step_y):
        for x in range(0, image_width - patch_width + 1, step_x):
            patch = patches[patch_idx]
            deconvolved_patch = deconvolved_patches[patch_idx]
            w1, w2 = w12vals[patch_idx]
            cropp = 12
            cropped_deconvolved_patch = deconvolved_patch[cropp:-cropp, cropp:-cropp]
            I_0_patch = cropped_deconvolved_patch * w2
            I_90_patch = cropped_deconvolved_patch * w1

            # Blend the deconvolved patch into the reconstructed image
            reconstructed_image[y:y + patch_height, x:x + patch_width] += cropped_deconvolved_patch
            I_0_total[y:y + patch_height, x:x + patch_width] += I_0_patch
            I_90_total[y:y + patch_height, x:x + patch_width] += I_90_patch

            # Blend the w1 and w2 values
            w1_map[y:y + patch_height, x:x + patch_width] += w1
            w2_map[y:y + patch_height, x:x + patch_width] += w2
            
            w_diff_map[y:y + patch_height, x:x + patch_width] += np.abs(w1 - w2)

            # Update the weight map for averaging
            weight_map[y:y + patch_height, x:x + patch_width] += 1

            patch_idx += 1

    # Normalize the reconstructed image, w1, w2 maps, and intensity maps by the weight map
    reconstructed_image /= np.maximum(weight_map, 1)  # Prevent division by zero
    w1_map /= np.maximum(weight_map, 1)
    w2_map /= np.maximum(weight_map, 1)
    w_diff_map /= np.maximum(weight_map, 1)
    I_0_total /= np.maximum(weight_map, 1)
    I_90_total /= np.maximum(weight_map, 1)

    # Calculate the final angle and magnitude maps from the blended w1 and w2 values
    angle_map = np.degrees(np.arctan2(w2_map, w1_map))
    magnitude_spectrum = np.sqrt(w1_map**2 + w2_map**2)

    num_rows = image_height // patch_height
    num_cols = image_width // patch_width
    patches_row_width = patch_width * num_cols
    patches_col_height = patch_height * num_rows

    I_0_total = I_0_total[:patches_col_height, :patches_row_width]
    I_90_total = I_90_total[:patches_col_height, :patches_row_width]
    angle_map = angle_map[:patches_col_height, :patches_row_width]
    magnitude_spectrum = magnitude_spectrum[:patches_col_height, :patches_row_width]
    reconstructed_image = reconstructed_image[:patches_col_height, :patches_row_width]
    w_diff_map = w_diff_map[:patches_col_height, :patches_row_width]
    w1_map = w1_map[:patches_col_height, :patches_row_width]
    w2_map = w2_map[:patches_col_height, :patches_row_width]


    # Plot the angle map, magnitude spectrum, and intensity channels
    plt.figure()
    plt.imshow(reconstructed_image, cmap='gray', alpha=0.5)
    plt.imshow(w1_map, alpha=0.5)
    plt.title("W1 Map")
    plt.colorbar()
    
    plt.figure()
    plt.imshow(reconstructed_image, cmap='gray', alpha=0.5)
    plt.imshow(angle_map, cmap='jet', alpha=0.5)
    plt.title(f"Angle Map channel {channel}")
    plt.colorbar()
    
    plt.figure()
    plt.imshow(reconstructed_image, cmap='gray', alpha=0.5)
    plt.imshow(magnitude_spectrum, cmap='jet', alpha=0.5)
    plt.title(f"Magnitude Spectrum channel {channel}")
    plt.colorbar()

    plt.figure()
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f"Reconstructed Image channel {channel}")
    
    plt.figure()
    plt.title(f"constructued times angle")
    plt.imshow(reconstructed_image * w_diff_map, cmap='jet')
    plt.colorbar()
    
    # plt.figure()
    # plt.title(f"constructued times magnitude")
    # plt.imshow(reconstructed_image * magnitude_spectrum, cmap='jet')
    # plt.colorbar()
    
    return reconstructed_image, angle_map, magnitude_spectrum

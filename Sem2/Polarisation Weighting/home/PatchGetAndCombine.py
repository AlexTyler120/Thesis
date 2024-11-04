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
from gt import gt
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
    step_y = int(patch_height * 0.45)
    step_x = int(patch_width * 0.45)
    # padding_size = patch_width//2
    padding_size = 6
    # convert bgr to rgb
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Loop over the image to extract patches with 50% overlap
    for y in range(0, img_height - patch_height + 1, step_y):
        for x in range(0, img_width - patch_width + 1, step_x):
            if (y + patch_height <= img_height) and (x + patch_width <= img_width):
                
                patch = image[y:y + patch_height, x:x + patch_width]
                # padded_patch = patch
                padded_patch = cv2.copyMakeBorder(
                    patch, padding_size, padding_size, padding_size, padding_size, 
                    borderType=cv2.BORDER_REFLECT
                    # borderType=cv2.BORDER_REPLICATE
                    # borderType=cv2.BORDER_CONSTANT
                )
                # plt.figure()
                # plt.subplot(1, 2, 1)
                # plt.imshow(patch)
                # plt.title("Original Patch")
                # plt.subplot(1, 2, 2)
                # plt.title("Padded Patch")
                # plt.imshow(padded_patch)
                # plt.show()
                patches.append(padded_patch)
    return patches
def tukey_window(N, alpha=0.5):
    """
    Create a 1D Tukey window of length N and taper ratio alpha.
    """
    # Create the Tukey window
    window = np.ones(N)
    
    # Handle the flat part of the window (center)
    for n in range(int(alpha * (N - 1) / 2)):
        window[n] = 0.5 * (1 + np.cos(np.pi * (2 * n / (alpha * (N - 1)) - 1)))
        window[N - n - 1] = window[n]
    
    return window
import numpy.ma as ma
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
    # Create Hanning window
    # hanning_window = np.outer(np.hanning(patch_height), np.hanning(patch_width))
    # hanning_window = np.outer(np.hamming(patch_height), np.hamming(patch_width))
    # alpha = 0.4  # Adjust alpha to control the amount of tapering
    # hanning_window = np.outer(tukey_window(patch_height, alpha), tukey_window(patch_width, alpha))
    # Create empty arrays for reconstructed images, angle, magnitude, and weight maps

    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)
    recon_other = np.zeros((image_height, image_width), dtype=np.float32)
    I_0_total = np.zeros_like(reconstructed_image)
    I_90_total = np.zeros_like(reconstructed_image)
    angle_map = np.zeros_like(reconstructed_image)
    magnitude_spectrum = np.zeros_like(reconstructed_image)
    w1_map = np.zeros_like(reconstructed_image)
    w2_map = np.zeros_like(reconstructed_image)
    w_diff_map = np.zeros_like(reconstructed_image)
    weight_map = np.zeros((image_height, image_width), dtype=np.float32)
    std_map = np.zeros((image_height, image_width), dtype=np.float32)
    patch_idx = 0
    step_y = int(patch_height * 0.45)
    step_x = int(patch_width * 0.45)
    
    num_cols = (image_width - patch_width) // step_x + 1
    num_rows = (image_height - patch_height) // step_y + 1
    
    # Loop over the patches and place them in their respective positions, averaging overlapping regions
    for y in range(0, image_height - patch_height + 1, step_y):
        for x in range(0, image_width - patch_width + 1, step_x):
            patch = patches[patch_idx]
            deconvolved_patch = deconvolved_patches[patch_idx] 
            
            w1, w2 = w12vals[patch_idx]
            decon_other = WeightingEstimate.deconvolve_img(patch[:,:, 0], WeightingEstimate.get_img_psf(w1, 5), wiener=True)
            cropp = 6
            cropped_deconvolved_patch = deconvolved_patch[cropp:-cropp, cropp:-cropp] #* hanning_window
            decon_cropped = decon_other[cropp:-cropp, cropp:-cropp]
            
        

            # Blend the deconvolved patch into the reconstructed image
            reconstructed_image[y:y + patch_height, x:x + patch_width] += cropped_deconvolved_patch
            recon_other[y:y + patch_height, x:x + patch_width] += decon_cropped
            
            I_0_patch = cropped_deconvolved_patch * w2
            I_90_patch = cropped_deconvolved_patch * w1
            
            I_0_total[y:y + patch_height, x:x + patch_width] += I_0_patch
            I_90_total[y:y + patch_height, x:x + patch_width] += I_90_patch
            # if w1 < w2:
            #     w1 = -w1
            # Blend the w1 and w2 values
            w1_map[y:y + patch_height, x:x + patch_width] += w1
            
            w2_map[y:y + patch_height, x:x + patch_width] += w2 
            
            std_map[y:y + patch_height, x:x + patch_width] += np.std(cropped_deconvolved_patch)

            # Update the weight map for averaging
            weight_map[y:y + patch_height, x:x + patch_width] += 1
            
            prev_deconvolved_patch = cropped_deconvolved_patch
            prev_w12_vals = (w1, w2)
            # weight_map[y:y + patch_height, x:x + patch_width] += hanning_window

            patch_idx += 1

    # Normalize the reconstructed image, w1, w2 maps, and intensity maps by the weight map
    reconstructed_image /= np.maximum(weight_map, 1e-5)  # Prevent division by zero
    w1_map /= np.maximum(weight_map, 1e-5)
    w2_map /= np.maximum(weight_map, 1e-5)
    w_diff_map /= np.maximum(weight_map, 1e-5)
    I_0_total /= np.maximum(weight_map, 1e-5)
    I_90_total /= np.maximum(weight_map, 1e-5)
    std_map /= np.maximum(weight_map, 1e-5)
    
    w_diff_map = w2_map - w1_map
    # normalse w1_map and w2_map
    global_min = min(np.min(w1_map), np.min(w2_map))
    global_max = max(np.max(w1_map), np.max(w2_map))
    w1_map = (w1_map - global_min) / (global_max - global_min)
    w2_map = (w2_map - global_min) / (global_max - global_min)
    
    # Calculate the final angle and magnitude maps from the blended w1 and w2 values
    angle_map = np.degrees(np.arctan2(w2_map, w1_map))
    # normalise 0 - 90
    angle_map = cv2.normalize(angle_map, None, 0, 90, cv2.NORM_MINMAX)
    
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
    
    std_map = std_map[:patches_col_height, :patches_row_width]
    # angle_map = np.clip(angle_map, 0, 55)
    # angle_map = cv2.normalize(angle_map, None, 0, 90, cv2.NORM_MINMAX)
    # normalise magnitude spectrum
    # magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))
    # angle_map *= magnitude_spectrum
    # angle_map = (angle_map - np.min(angle_map)) / (np.max(angle_map) - np.min(angle_map))
    
    # angle_map = np.clip(angle_map, 0, 33)
    # angle_map = 33 - angle_map
    # Plot the angle map, magnitude spectrum, and intensity channels
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(I_0_total, cmap="jet")
    # plt.title("I_0")
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(I_90_total, cmap="jet")
    # plt.title("I_90")
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(reconstructed_image, cmap='gray', alpha=0.5)
    # plt.imshow(w1_map, alpha=0.5)
    # plt.title("W1 Map")
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(reconstructed_image, cmap='gray', alpha=0.5)
    # plt.imshow(w2_map, alpha=0.5)
    # plt.title("W2 Map")
    # plt.colorbar()
    plt.figure()
    plt.imshow(reconstructed_image, cmap='gray', alpha=0.5)
    plt.imshow(angle_map, cmap='jet', alpha=0.5)#, vmin = 0, vmax = 90)
    plt.title(f"Angle Map channel {channel}")
    plt.colorbar()
    
    # plt.figure()
    # plt.imshow(reconstructed_image, cmap='gray', alpha=0.5)
    # guess = angle_map * magnitude_spectrum
    # max_g = np.max(guess)
    # plt.imshow(guess, cmap='jet', alpha=0.5)#, vmin = 0, vmax = 90)
    # plt.title(f"Angle Map channe times l {channel}")
    # plt.colorbar()
    
    plt.figure()
    plt.imshow(reconstructed_image, cmap='gray', alpha=0.5)
    plt.imshow(magnitude_spectrum, cmap='jet', alpha=0.5)
    plt.title(f"Magnitude Spectrum channel {channel}")
    plt.colorbar()

    plt.figure()
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f"Reconstructed Image channel {channel}")
    
    # plt.figure()
    # plt.imshow(std_map, cmap='jet')
    # plt.title(f"Standard Deviation Map channel {channel}")
    
    polar_angles, dolp, ca, cd = gt()
    
    ca = ca[:patches_col_height, :patches_row_width]
    cd = cd[:patches_col_height, :patches_row_width]
    
    # Threshold for displaying the angle map
    threshold = 0.3

    # Create a mask for pixels where dolp >= threshold
    if channel == 0:
        dolp_c = dolp['R']
    elif channel == 1:
        dolp_c = dolp['G']
    else:
        dolp_c = dolp['B']
    dolp_f = dolp_c[:patches_col_height, :patches_row_width]
    
    mask = ma.masked_where(dolp_f < threshold, angle_map)

    # Create a new array to display, which will show the reconstructed image where mask is False

    # Plotting the result
    plt.figure()
    plt.imshow(reconstructed_image, cmap='gray')  # Underlay the reconstructed image
    plt.imshow(mask, cmap='jet')  # Overlay the angle map where dolp >= threshold
    plt.colorbar(label='Angle Map Intensity')  # Only shows the colorbar for the angle map
    plt.title(f"Angle Map Masked channel {channel}")
    plt.figure()
    if channel == 0:
        polarb = polar_angles['R']
    elif channel == 1:
        polarb = polar_angles['G']
    else:
        polarb = polar_angles['B']
    polarb = polarb[:patches_col_height, :patches_row_width]
    # mask polar b to wher edolp_f is greater than threshold
    mask = ma.masked_where(dolp_f < threshold, polarb)
    display_image = np.where(mask, polarb, reconstructed_image)
    plt.imshow(mask, cmap='jet')  # Overlay the angle map where dolp >= threshold
    plt.colorbar(label='Angle Map Intensity')  # Only shows the colorbar for the angle map
    plt.title(f"Ground Truth AoLP Masked channel {channel}")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(dolp_c, cmap='jet')
    plt.title(f'DoLP channel {channel}')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(polarb, cmap='jet')
    plt.title(f'AoLP channel {channel}')
    plt.colorbar()
    if channel == 2:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(ca, cmap='jet')
        plt.title(f'Composite AoLP')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(cd, cmap='jet')
        plt.title(f'Composite DoLP')
        plt.colorbar()
        
        
    return reconstructed_image, angle_map, magnitude_spectrum, cd, ca

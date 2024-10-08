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
def extract_image_patch_overlap(image, patch_size, overlap=0.5):
    """_summary_

    Args:
        image (_type_): _description_
        patch_size (_type_): _description_
        overlap (float, optional): _description_. Defaults to 0.5.
    """
    patches = []
    img_height, img_width = image.shape[:2]
    patch_height, patch_width = patch_size
    
    # Calculate the stride for the given overlap
    stride_height = int(patch_height * (1 - overlap))
    stride_width = int(patch_width * (1 - overlap))
    
    # Loop over the image to extract patches 
    for y in range(0, img_height - patch_height + 1, stride_height):
        for x in range(0, img_width - patch_width + 1, stride_width):
            patch = image[y:y+patch_height, x:x+patch_width]
            patches.append(patch)
            
    return patches

def create_cosine_window(patch_size):
    """Create a cosine window for blending."""
    patch_height, patch_width = patch_size
    # Create a 1D cosine window
    y_window = np.hanning(patch_height)
    x_window = np.hanning(patch_width)
    # Create a 2D window by multiplying the 1D windows
    window = np.outer(y_window, x_window)
    return window

def reconstruct_image_from_patches_overlap(patches, image_size, patch_size, overlap=0.5):
    """_summary_

    Args:
        patches (_type_): _description_
        image_size (_type_): _description_
        patch_size (_type_): _description_
        overlap (float, optional): _description_. Defaults to 0.5.
    """
    # return reconstructed_image
    image_height, image_width = image_size
    patch_height, patch_width = patch_size
    
    # Calculate the stride for the given overlap
    stride_height = int(patch_height * (1 - overlap))
    stride_width = int(patch_width * (1 - overlap))
    
    # Create an empty array for the reconstructed image
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)
    # Create an empty array to keep track of the sum of weights
    weight_sum = np.zeros((image_height, image_width), dtype=np.float32)
    
    # Create the cosine window
    window = create_cosine_window(patch_size)
    
    patch_idx = 0
    # Loop over the image to place patches in the correct position
    for y in range(0, image_height - patch_height + 1, stride_height):
        for x in range(0, image_width - patch_width + 1, stride_width):
            # Apply the window to the patch
            weighted_patch = patches[patch_idx] * window
            # Place the weighted patch back into the image
            reconstructed_image[y:y+patch_height, x:x+patch_width] += weighted_patch
            # Add the window weights to the weight_sum
            weight_sum[y:y+patch_height, x:x+patch_width] += window
            patch_idx += 1
    
    # Normalize the reconstructed image by the weight_sum to account for overlap
    reconstructed_image /= weight_sum
    # Clip the values to be in valid range
    reconstructed_image = np.clip(reconstructed_image, 0, 255)
    
    return reconstructed_image

def reconstruct_image_from_patches_overlap_with_quiver(patches, image_size, patch_size, w12_vals, channel, overlap=0.5):
    """Reconstruct image from patches with advanced blending using cosine window and add quiver plot.
    
    Args:
        patches (numpy.ndarray): Array of patches to reconstruct the image from.
        image_size (tuple): Size of the original image (height, width).
        patch_size (tuple): Size of each patch (height, width).
        w12_vals (list of tuples): List of (U, V) tuples for quiver plot corresponding to each patch.
        channel (int): Channel index to specify quiver arrow color.
        shift (int): Padding size to be removed from patches.
        overlap (float): Overlap ratio between patches. Defaults to 0.5.
    
    Returns:
        numpy.ndarray: Reconstructed image with quiver plot.
    """
    image_height, image_width = image_size
    patch_height, patch_width = patch_size
    
    # Calculate the stride for the given overlap
    stride_height = int(patch_height * (1 - overlap))
    stride_width = int(patch_width * (1 - overlap))
    
    # Create an empty array for the reconstructed image
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)
    # Create an empty array to keep track of the sum of weights
    weight_sum = np.zeros((image_height, image_width), dtype=np.float32)
    
    # Create the cosine window
    window = create_cosine_window(patch_size)
    
    patch_idx = 0
    X_quiver = []
    Y_quiver = []
    U_quiver = []
    V_quiver = []
    
    # Loop over the image to place patches in the correct position
    for y in range(0, image_height - patch_height + 1, stride_height):
        for x in range(0, image_width - patch_width + 1, stride_width):
            # Apply the window to the patch
            weighted_patch = patches[patch_idx] * window
            
            # Place the weighted patch back into the image
            reconstructed_image[y:y+patch_height, x:x+patch_width] += weighted_patch
            # Add the window weights to the weight_sum
            weight_sum[y:y+patch_height, x:x+patch_width] += window
            
            # Get the quiver dx, dy values for this patch
            U, V = w12_vals[patch_idx]  # Assuming w12_vals contains (U, V) for each patch
            
            # Calculate the center of the current patch
            center_x = x + patch_width // 2
            center_y = y + patch_height // 2
            
            # Store the center and the quiver values
            X_quiver.append(center_x)
            Y_quiver.append(center_y)
            U_quiver.append(U)
            V_quiver.append(V)
            
            patch_idx += 1
    
    # Normalize the reconstructed image by the weight_sum to account for overlap
    reconstructed_image /= weight_sum
    # Clip the values to be in valid range
    reconstructed_image = np.clip(reconstructed_image, 0, 255)
    
    # Determine the color of the quiver arrows based on the channel
    if channel == 0:
        colour = 'red'
    elif channel == 1:
        colour = 'green'
    elif channel == 2:
        colour = 'blue'
    else:
        colour = 'black'    

    # Plot the reconstructed image with quiver arrows
    plt.figure(figsize=(10, 10))
    plt.imshow(reconstructed_image, cmap='gray')
    plt.quiver(X_quiver, Y_quiver, U_quiver, V_quiver, color=colour, angles='xy', scale_units='xy', scale=0.1)
    plt.title(f"Reconstructed Image with Quiver Arrows for channel {colour}")
    
    return reconstructed_image

def extract_image_patches_no_overlap(image, patch_size, shift):
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
    padding_size = shift

    # Loop over the image to extract patches, ignoring smaller patches at edges
    for y in range(0, img_height, patch_height):
        for x in range(0, img_width, patch_width):
            # Ensure patch size is valid (no smaller patches at the edges)
            if (y + patch_height <= img_height) and (x + patch_width <= img_width):
                
                patch = image[y:y+patch_height, x:x+patch_width]
                # Now add the padding from the surrounding image
                y_start_pad = max(0, y - padding_size)
                y_end_pad = min(img_height, y + patch_height + padding_size)
                x_start_pad = max(0, x - padding_size)
                x_end_pad = min(img_width, x + patch_width + padding_size)

                # Extract the larger region around the patch
                padded_patch = image[y_start_pad:y_end_pad, x_start_pad:x_end_pad]

                # Check if we are at the edge and if extra padding is needed
                # Use BORDER_REPLICATE to fill the missing areas if we're at the boundary
                if padded_patch.shape[0] < patch_height + 2 * padding_size or padded_patch.shape[1] < patch_width + 2 * padding_size:
                    padded_patch = cv2.copyMakeBorder(
                        padded_patch,
                        top=padding_size - (y - y_start_pad), bottom=padding_size - (y_end_pad - (y + patch_height)),
                        left=padding_size - (x - x_start_pad), right=padding_size - (x_end_pad - (x + patch_width)),
                        # borderType=cv2.BORDER_REPLICATE
                        borderType=cv2.BORDER_REFLECT
                    )
                    
                # padded_patch = cv2.copyMakeBorder(
                #     patch, padding_size, padding_size, padding_size, padding_size, 
                #     # borderType=cv2.BORDER_REFLECT
                #     # borderType=cv2.BORDER_REPLICATE
                #     # borderType=cv2.BORDER_CONSTANT
                # )
                patches.append(padded_patch)
                # patches.append(patch)
    
    return patches


def reconstruct_image_from_patches_no_overlap(patches, image_size, patch_size, shift):
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
    padding_size = shift
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

def colour_stuff_filtering(angle_map, reconstructed_image, channel, image_width, image_height, valid_mask):
    # Normalisation of the angle map
    normalised_angle_map = angle_map / np.max(angle_map)

    # Resize the normalised angle map for visualisation
    angle_map_resized = cv2.resize(normalised_angle_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

    # Apply Gaussian filtering to the clustered angle map
    gaussian_filtered_map = gaussian_filter(normalised_angle_map, sigma=1)

    mid_size = 3
    # Apply median filtering to the clustered angle map
    median_filtered_map = median_filter(normalised_angle_map, size=mid_size)
    
    more_size = 4
    median_filter_more = median_filter(normalised_angle_map, size=more_size)
    less_size = 2
    median_filter_less = median_filter(normalised_angle_map, size=less_size)
    # Normalise the filtered angle maps for visualisation
    maps = {
        'Original': angle_map_resized,
        'Gaussian Filtered': gaussian_filtered_map,
        f'Median Filtered size {mid_size}': median_filtered_map,
        f"Median Filter size {more_size}": median_filter_more,
        f"Median Filter size {less_size}": median_filter_less
    }

    # Resize and create colour overlays for each filtered map
    colour_overlays = {}
    for key, filtered_map in maps.items():
        normalised_map = filtered_map / np.max(filtered_map)
        resized_map = cv2.resize(normalised_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
        # colour_overlays[key] = (plt.cm.jet(resized_map)[:, :, :3] * 255).astype(np.uint8)
        colour_overlay = (plt.cm.jet(resized_map)[:, :, :3] * 255).astype(np.uint8)
        valid_mask_resized = cv2.resize(valid_mask.astype(np.uint8), (image_width, image_height), interpolation=cv2.INTER_NEAREST).astype(bool)
        colour_overlay[~valid_mask_resized] = cv2.cvtColor(reconstructed_image, cv2.COLOR_GRAY2RGB)[~valid_mask_resized]

        # colour_overlay[~valid_mask] = cv2.cvtColor(reconstructed_image[~valid_mask], cv2.COLOR_GRAY2RGB)[~valid_mask]
        colour_overlays[key] = colour_overlay


    # Normalise and convert reconstructed grayscale image to 3-channel RGB
    grayscale_image_normalised = cv2.normalize(reconstructed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grayscale_image_3c = cv2.cvtColor(grayscale_image_normalised, cv2.COLOR_GRAY2RGB)

    # Combine grayscale image with different overlays
    combined_images = {}
    for key, overlay in colour_overlays.items():
        combined_images[key] = cv2.addWeighted(grayscale_image_3c, 0.5, overlay, 0.5, 0)

    # Plotting the images in a 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(30, 20))  # Adjusted figsize for 2x3 subplots
    axs = axs.flatten()

    # Display the original and filtered maps with gradient overlays
    for ax, (title, combined_image) in zip(axs, combined_images.items()):
        ax.imshow(combined_image)
        ax.set_title(f"{title} Angle Map for Channel {channel}")
        ax.axis('off')

    # Display the reconstructed image
    axs[-1].imshow(grayscale_image_normalised, cmap='gray')
    axs[-1].set_title("Reconstructed Image")
    axs[-1].axis('off')

    # Create a colourbar for angle values that spans all subplots
    fig.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)  # Adjust hspace and wspace as needed
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # Position of colourbar [left, bottom, width, height]
    norm = plt.Normalize(vmin=0, vmax=45)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Angle (degrees)")
    

def reconstruct_image_from_patches_no_overlap_with_quiver(patches, image_size, patch_size, w12_vals, channel, shift):
    """
    Reconstructs an image from patches and adds a single quiver arrow for each patch based on dx, dy values.
    Clusters angles and sets noisy regions to 0 degrees.
    
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
    padding_size = shift
    
    # Create an empty array for the reconstructed image
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)

    patch_idx = 0
    X_quiver = []
    Y_quiver = []
    U_quiver = []
    V_quiver = []
    
    angles = []
    std_devs = []
    angle_map = np.zeros((image_height // patch_height, image_width // patch_width))
    std_map = np.zeros((image_height // patch_height, image_width // patch_width))
    valid_mask = np.zeros((image_height, image_width), dtype=bool)
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
                w1, w2 = w12_vals[patch_idx]
                if not (0.05 <= w1 <= 0.48 or 0.52 <= w1 <= 0.95):
                    valid_mask[y:y+patch_height, x:x+patch_width] = True
                else:
                    valid_mask[y:y+patch_height, x:x+patch_width] = True
                # Get the quiver dx, dy values for this patch
                U, V = w12_vals[patch_idx]  # Assuming w12_vals contains (U, V) for each patch

                # Calculate the center of the current patch
                center_x = x + patch_width // 2
                center_y = y + patch_height // 2

                # Store the center and the quiver values
                X_quiver.append(center_x)
                Y_quiver.append(center_y)
                U_quiver.append(U)
                V_quiver.append(V)
                
                angle = np.arctan2(V, U)
                # print(f"Angle: {np.degrees(angle)}")
                # get std of image
                std_dev = np.std(cropped_patch)
                if std_dev < 0.012:
                    angle = 0
                std_dev = np.std(patch)
                std_devs.append(std_dev)
                std_map[y // patch_height, x // patch_width] = std_dev

                    
                angles.append(angle)
                angle_map[y // patch_height, x // patch_width] = np.degrees(angle)
                
                
                patch_idx += 1
        
    colour_stuff_filtering(angle_map, reconstructed_image, channel, image_width, image_height, valid_mask)

    return reconstructed_image

def reconstruct_image_patch_intensity(patches, deconvolved_patches, image_size, patch_size, shift, channel, w12vals):
    image_height, image_width = image_size
    patch_height, patch_width = patch_size
    padding_size = shift
    valid_mask = np.zeros((image_height, image_width), dtype=bool)  # Mask to store valid patches

    # Create an empty array for the reconstructed image
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)
    angle_map = np.zeros((image_height // patch_height, image_width // patch_width))
    angle_image = np.zeros((image_height, image_width), dtype=np.float32)
    angle_patches = np.zeros((image_height // patch_height, image_width // patch_width))
    patch_idx = 0
    dop_proxy = np.zeros_like(reconstructed_image)
    I_0_total = np.zeros_like(reconstructed_image)
    I_90_total = np.zeros_like(reconstructed_image)
    aop_map = np.zeros_like(reconstructed_image)
    dop_map = np.zeros_like(reconstructed_image)
    # Loop over the image to place patches in the correct position
    for y in range(0, image_height, patch_height):
        for x in range(0, image_width, patch_width):
            # Ensure we are not placing smaller patches (ignored at edges)
            if (y + patch_height <= image_height) and (x + patch_width <= image_width):
                # Remove the padding by cropping the padding from all sides
                patch = patches[patch_idx]
                patch = patch[:,:, channel]
                deconvolved_patch = deconvolved_patches[patch_idx]
                w1, w2 = w12vals[patch_idx]
                if not (0.05 <= w1 <= 0.48 or 0.52 <= w1 <= 0.95):
                    valid_mask[y:y+patch_height, x:x+patch_width] = True
                else:
                    valid_mask[y:y+patch_height, x:x+patch_width] = True
                    # patch_idx += 1
                    # continue  # Skip this patch if the condition is not met
                
                cropped_patch = patch[padding_size:-padding_size, padding_size:-padding_size]
                cropped_deconvolved_patch = deconvolved_patch[padding_size:-padding_size, padding_size:-padding_size]
                # Place the cropped patch back into the image
                reconstructed_image[y:y+patch_height, x:x+patch_width] = cropped_deconvolved_patch
                if np.std(cropped_patch) < 0.0125 or not (0.05 <= w1 <= 0.48 or 0.52 <= w1 <= 0.95):
                    intensity_diff = 0
                else:
                    intensity_diff = np.abs(np.median(cropped_deconvolved_patch + cropped_patch))
                # intensity_diff = np.degrees(np.arctan2(w1, 1)) * np.abs(np.median(cropped_deconvolved_patch - cropped_patch))
                
                # intensity_diff = np.arctan2(pos_diff + 1e-8, neg_diff + 1e-8)
                intensity_diff = np.degrees(np.arctan2(w2, w1))
                intensity_diff = w1
                min_angle, max_angle = 0.0, 45.0
                # if w1 > 0.5:
                #     intensity_diff /= w1
                # else:
                #     intensity_diff /= w2
                polarisation_angle = intensity_diff

                # angles for each pixel
                angles = np.zeros_like(cropped_deconvolved_patch)
                I_0_patch = np.zeros_like(cropped_deconvolved_patch)
                I_90_patch = np.zeros_like(cropped_deconvolved_patch)
                aop_patch = np.zeros_like(cropped_deconvolved_patch)
                dop_patch = np.zeros_like(cropped_deconvolved_patch)
                for i in range(patch_height):
                    for j in range(patch_width):
                        angles[i,j] = (w1 - w2) / (w2 + w1)
                        angles[i,j] = np.clip(angles[i,j], -0.99, 0.99)
                        I_0_patch[i,j] = cropped_deconvolved_patch[i,j] / (1 + angles[i,j])
                        I_90_patch[i,j] = cropped_deconvolved_patch[i,j] / (1 - angles[i,j])
                        I_0_patch[i,j] = (I_0_patch[i,j] - np.min(I_0_patch)) / (np.max(I_0_patch) - np.min(I_0_patch))
                        I_90_patch[i,j] = (I_90_patch[i,j] - np.min(I_90_patch)) / (np.max(I_90_patch) - np.min(I_90_patch))
                        if np.std(cropped_deconvolved_patch) < 0.02:
                            aop_patch[i,j] = 0
                        else:
                            aop_patch[i,j] = np.degrees(0.5 * np.arctan2(np.abs(I_90_patch[i,j] - I_0_patch[i,j]), I_0_patch[i,j] + I_90_patch[i,j]))
                        dop_patch[i,j] = np.abs(I_0_patch[i,j] - I_90_patch[i,j]) / (I_0_patch[i,j] + I_90_patch[i,j] + 1e-8)
                
                angle_image[y: y + patch_height, 
                            x : x + patch_width] = angles
                
                # Place the cropped patch back into the image
                reconstructed_image[y:y+patch_height, x:x+patch_width] = cropped_deconvolved_patch
                I_0_total[y:y+patch_height, x:x+patch_width] = I_0_patch
                I_90_total[y:y+patch_height, x:x+patch_width] = I_90_patch
                aop_map[y:y+patch_height, x:x+patch_width] = aop_patch
                # Store the polarisation angle in the angle map
                angle_map[y // patch_height, x // patch_width] = polarisation_angle
                dop_proxy[y:y+patch_height, x:x+patch_width] = (w1 - w2) / (w2 + w1) # -1 w90 much stronger dominance of 90 deg polarisation 
                dop_map[y:y+patch_height, x:x+patch_width] = dop_patch
                # 1 w0 much stronger dominance of 0 deg polarisation
                patch_idx += 1
    print(f"Min angle: {np.min(angle_map)}, Max angle: {np.max(angle_map)}")
    # normalise angle map 0 -45
    colour_stuff_filtering(aop_map, reconstructed_image, 0, image_width, image_height, valid_mask)
    plt.figure()
    plt.imshow(I_0_total, cmap='jet')
    plt.title("I_0")
    plt.colorbar(label='DoP Proxy')
    plt.figure()
    plt.imshow(I_90_total, cmap='jet')
    plt.title("I_90")
    plt.colorbar(label='DoP Proxy')
    plt.figure()
    plt.imshow(aop_map, cmap='jet', alpha=0.6)
    plt.imshow(reconstructed_image, cmap='gray', alpha=0.4)
    plt.colorbar(label='AoP')
    plt.title("AoP")
    plt.figure()
    plt.imshow(dop_map, cmap='jet')
    plt.colorbar(label='DoP')
    
    
    # plt.colorbar(label='Angle (degrees)')
    return reconstructed_image



def create_full_quiver(image, image_size, patch_size, w12_vals):
    image_height, image_width = image_size
    patch_height, patch_width = patch_size
    patch_idx = 0
    X_quiver = []
    Y_quiver = []
    U_quiver = []
    V_quiver = []
    
    angles = []
    std_devs = []
    angle_map = np.zeros((image_height // patch_height, image_width // patch_width))
    std_map = np.zeros((image_height // patch_height, image_width // patch_width))
    for y in range(0, image_height, patch_height):
        for x in range(0, image_width, patch_width):
            if (y + patch_height <= image_height) and (x + patch_width <= image_width):
                # Get the quiver dx, dy values for this patch
                U, V = w12_vals[patch_idx]  # Assuming w12_vals contains (U, V) for each patch
                    
                # Calculate the center of the current patch
                center_x = x + patch_width // 2
                center_y = y + patch_height // 2

                # Store the center and the quiver values
                X_quiver.append(center_x)
                Y_quiver.append(center_y)
                U_quiver.append(U)
                V_quiver.append(V)
                                
                angle = np.arctan2(U, V)
                angles.append(angle)
                angle_map[y // patch_height, x // patch_width] = np.degrees(angle)
                
                # Calculate standard deviation of the RGB intensity
                patch = image[y:y + patch_height, x:x + patch_width]
                std_dev = np.std(patch)
                std_devs.append(std_dev)
                std_map[y // patch_height, x // patch_width] = std_dev
                
                patch_idx += 1
                
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.quiver(X_quiver, Y_quiver, U_quiver, V_quiver, color='black', angles='xy', scale_units='xy', scale=0.1)
    plt.title(f"Reconstructed Image with Single Quiver Arrow per Patch for rgb")
    
    # Plot the angle colormap
    plt.figure(figsize=(10, 10))
    plt.imshow(angle_map, cmap='hsv', extent=(0, image_width, image_height, 0), alpha = 1)
    plt.colorbar(label="Angle (deg)")
    plt.title(f"Angle Colormap for Each Patch")
    
    normalized_angle_map = (angle_map - np.min(angle_map)) / (np.max(angle_map) - np.min(angle_map))
    # Calculate the 75th percentile of the standard deviations
    std_threshold = np.percentile(std_devs, 60)
    std_map_weighted = np.where(std_map >= std_threshold, std_map, 0)
    normalized_std_map = (std_map_weighted - np.min(std_map_weighted)) / (np.max(std_map_weighted) - np.min(std_map_weighted) + 1e-5)
    # normalized_std_map = (std_map - np.min(std_map)) / (np.max(std_map) - np.min(std_map))
    
    angle_map_resized = cv2.resize(normalized_angle_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    std_map_resized = cv2.resize(normalized_std_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    
    # Convert the original image to HSV
    hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    
    # Adjust the hue channel of the HSV image based on the angle map
    hsv_image[..., 0] = (angle_map_resized * 179).astype(np.uint8)  # Hue values range from 0 to 179
    # Adjust the saturation channel based on the standard deviation map
    # hsv_image[..., 1] = (std_map_resized * 255).astype(np.uint8)  # Saturation values range from 0 to 255
    
    # Convert the image back to RGB
    combined_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Display the combined image
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(combined_image)
    plt.title(f"Original Image Adjusted by the Angle Colormap")
    plt.axis('off')
    # Create a colorbar for hue (angle in degrees)
    norm = plt.Normalize(vmin=0, vmax=45)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hue (Angle in degrees)")
    plt.show()
    
    # Reshape the angle map for clustering
    angle_map_reshaped = angle_map.reshape(-1, 1)

    # Apply K-means clustering to the angles
    kmeans = KMeans(n_clusters=4, random_state=0).fit(angle_map_reshaped)
    clustered_angles = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape clustered angles back to the original angle map shape
    clustered_angle_map = clustered_angles.reshape(angle_map.shape)

    # Apply Gaussian filtering to smooth the clustered angle map
    smooth_clustered_angle_map = gaussian_filter(clustered_angle_map, sigma=1)  # Adjust sigma for more/less smoothing

    # Normalize clustered angle map to [0, 1] range
    normalized_smooth_clustered_angle_map = (smooth_clustered_angle_map - np.min(smooth_clustered_angle_map)) / (np.max(smooth_clustered_angle_map) - np.min(smooth_clustered_angle_map))
    
    # Resize angle map to match the original image size
    angle_map_resized = cv2.resize(normalized_smooth_clustered_angle_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    
    # Create a color gradient overlay from the angle map
    color_overlay = plt.cm.jet(angle_map_resized)[:, :, :3]  # Use 'jet' colormap, taking RGB channels
    
    # Normalize and convert to uint8
    grayscale_image_3c = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert color_overlay to uint8
    color_overlay = (color_overlay * 255).astype(np.uint8)

    # Combine the grayscale image and the color overlay
    combined_image = cv2.addWeighted(grayscale_image_3c, 0.55, color_overlay, 0.45, 0)
    
    # Display the combined image with gradient overlay
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(combined_image)
    plt.title("Grayscale Image with Smooth Clustered Angle Gradient Overlay")
    plt.axis('off')
    
    # Create a colorbar for angle values
    norm = plt.Normalize(vmin=0, vmax=90)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Angle (degrees)")
    
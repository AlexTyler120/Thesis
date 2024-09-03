import WeightingEstimate
import Autocorrelation
import ShiftEstimate
import matplotlib.pyplot as plt
import skimage as sk
import numpy as np

def run_estimate_w1_w2_patches(transformed_image, patch_size=(50, 50)):
    """
    Run estimation getting w1 and w2 for each patch.
    transformed_image: the image to estimate the weighting for.
    patch_size: the size of each patch.
    """
    # Get image dimensions
    img_height, img_width, num_channels = transformed_image.shape

    # Calculate padding
    pad_height = (patch_size[0] - (img_height % patch_size[0])) % patch_size[0]
    pad_width = (patch_size[1] - (img_width % patch_size[1])) % patch_size[1]

    # Apply padding if necessary
    if pad_height > 0 or pad_width > 0:
        transformed_image = np.pad(transformed_image, 
                                   ((0, pad_height), (0, pad_width), (0, 0)), 
                                   mode='constant', constant_values=0)

    # Now you can divide the image into blocks
    patches = sk.util.view_as_blocks(transformed_image, block_shape=(patch_size[0], patch_size[1], num_channels))

    deconvolved_patches = []

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0, :, :, :]
            deconvolved_patch_channels = []
            
            for channel in range(patch.shape[2]):
                patch_channel = patch[:, :, channel]
                # shift_estimation = ShiftEstimate.compute_pixel_shift(patch_channel)
                shift_estimation = 6  # Placeholder, replace with actual estimation if needed
                est_w1, est_w2 = WeightingEstimate.optimise_psf_both_weight(patch_channel, shift_estimation)
                deconvolved_patch_channel = deconvolve_patch(patch_channel, est_w1, est_w2, shift_estimation)
                deconvolved_patch_channels.append(deconvolved_patch_channel)
            
            # Combine the deconvolved channels back into an RGB patch
            deconvolved_patch = np.stack(deconvolved_patch_channels, axis=2)
            deconvolved_patches.append(deconvolved_patch)

    # Reconstruct the full image from patches
    reconstructed_image = reconstruct_from_patches(deconvolved_patches, transformed_image.shape)
    
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("Reconstructed Deconvolved Image")
    plt.show()

def deconvolve_patch(patch, w1, w2, shift_val):
    """
    Deconvolve a patch using the estimated weights and shift value
    patch: the patch to deconvolve
    w1, w2: estimated weights
    shift_val: estimated shift value
    """
    psf_patch = WeightingEstimate.get_img_psf_w1_w2(w1, w2, shift_val)
    deconvolved_patch = sk.restoration.wiener(patch, psf_patch, balance=1e-6)
    return deconvolved_patch

def reconstruct_from_patches(deconvolved_patches, original_shape):
    """
    Reconstruct the full image from deconvolved patches
    deconvolved_patches: list of deconvolved patches
    original_shape: the shape of the original image
    """
    patch_size = deconvolved_patches[0].shape
    reconstructed_image = np.zeros(original_shape)

    count = 0
    for i in range(0, original_shape[0], patch_size[0]):
        for j in range(0, original_shape[1], patch_size[1]):
            reconstructed_image[i:i+patch_size[0], j:j+patch_size[1], :] = deconvolved_patches[count]
            count += 1
    
    return reconstructed_image
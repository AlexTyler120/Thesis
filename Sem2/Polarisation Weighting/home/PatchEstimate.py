import WeightingEstimate
import Autocorrelation
import ShiftEstimate
import matplotlib.pyplot as plt
import skimage as sk
import numpy as np
from skimage.util import view_as_windows

def run_estimate_w1_w2_patches(transformed_image, patch_size=(75, 75), overlap=(25, 25)):
    """
    Run estimation getting w1 and w2 for each overlapping patch, ignoring patches that hit the edges.
    transformed_image: the image to estimate the weighting for.
    patch_size: the size of each patch.
    overlap: the overlap between patches.
    """
    img_height, img_width, num_channels = transformed_image.shape

    # Determine steps
    step_y = patch_size[0] - overlap[0]
    step_x = patch_size[1] - overlap[1]

    deconvolved_patches = []
    patch_positions = []

    total_patches_y = (img_height - patch_size[0]) // step_y + 1
    total_patches_x = (img_width - patch_size[1]) // step_x + 1

    for i_idx, i in enumerate(range(0, img_height - patch_size[0] + 1, step_y)):
        for j_idx, j in enumerate(range(0, img_width - patch_size[1] + 1, step_x)):
            # Define patch area, ensuring it doesn't hit the edges
            patch = transformed_image[i:i + patch_size[0], j:j + patch_size[1], :]

            deconvolved_patch_channels = []
            for channel in range(patch.shape[2]):
                print(f"Processing patch {i_idx + 1}/{total_patches_y}, {j_idx + 1}/{total_patches_x} of channel {channel + 1}/{num_channels}")
                patch_channel = patch[:, :, channel]
                # shift_estimation = ShiftEstimate.compute_pixel_shift(patch_channel)
                shift_estimation = 6  # Placeholder, replace with actual estimation if needed
                est_w1, est_w2 = WeightingEstimate.optimise_psf_both_weight(patch_channel, shift_estimation)
                deconvolved_patch_channel = deconvolve_patch(patch_channel, est_w1, est_w2, shift_estimation)
                deconvolved_patch_channels.append(deconvolved_patch_channel)

            deconvolved_patch = np.stack(deconvolved_patch_channels, axis=2)
            deconvolved_patches.append(deconvolved_patch)
            patch_positions.append((i, j))

    # Reconstruct the full image from patches with overlap handling
    reconstructed_image = reconstruct_from_patches_with_overlap(deconvolved_patches, patch_positions, transformed_image.shape)
    
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("Reconstructed Deconvolved Image")
    plt.show()

def deconvolve_patch(patch, w1, w2, shift_val):
    """
    Deconvolve a patch using the estimated weights and shift value.
    patch: the patch to deconvolve.
    w1, w2: estimated weights.
    shift_val: estimated shift value.
    """
    psf_patch = WeightingEstimate.get_img_psf_w1_w2(w1, w2, shift_val)
    deconvolved_patch = sk.restoration.wiener(patch, psf_patch, balance=1e-6)
    return deconvolved_patch

def reconstruct_from_patches_with_overlap(deconvolved_patches, patch_positions, original_shape):
    """
    Reconstruct the full image from deconvolved patches with overlap handling.
    deconvolved_patches: list of deconvolved patches.
    patch_positions: list of positions for each patch.
    original_shape: the shape of the original image.
    """
    reconstructed_image = np.zeros(original_shape)
    patch_count = np.zeros(original_shape)

    for patch, (i, j) in zip(deconvolved_patches, patch_positions):
        end_i = i + patch.shape[0]
        end_j = j + patch.shape[1]
        reconstructed_image[i:end_i, j:end_j, :] += patch
        patch_count[i:end_i, j:end_j, :] += 1

    # Normalize by the number of patches contributing to each pixel
    reconstructed_image /= np.maximum(patch_count, 1)

    return reconstructed_image

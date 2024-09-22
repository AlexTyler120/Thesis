import PatchGetAndCombine
import ImageRun
import ShiftEstimate
import cv2
import pickle
import matplotlib.pyplot as plt
import Viewer
import numpy as np
import patchify

def get_patches(img, patch):
    """
    Get patches from an image.
    
    """
    patches, patch_info, overlap = PatchGetAndCombine.seperate_imgs_into_patches(img, patch)
    print(f"Number of patches: {len(patches)}")
    print(f"Patch overlap: {overlap}")
    return patches, patch_info, overlap

def get_saved_patches(path, patches_len):
    """
    Get images from saved files
    """
    all_patches = []
    for i in range(patches_len):
        img = cv2.imread(f"{path}/patch_{i}.png", cv2.IMREAD_GRAYSCALE)
        all_patches.append(img)
        
    return all_patches

def gaussian_combine_patches(patches, patch_info, img_size, overlap):
    """
    Guassian Blur edges
    """
    return PatchGetAndCombine.gaussian_blend_patches(patches, patch_info, img_size, overlap)

def combine_to_rgb(final0, final1, final2):
    """Combine into RGB. Back to front because of BGR

    Args:
        final0 (_type_): _description_
        final1 (_type_): _description_
        final2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return cv2.merge([final2, final1, final0])

def combine_all_channels(patches, patch_info, img_size, overlap, path):
    """_summary_

    Args:
        patches (_type_): _description_
        patch_info (_type_): _description_
        img_size (_type_): _description_
        overlap (_type_): _description_
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    all_patch_r = get_saved_patches(f"{path}/channel0_new", len(patches))
    all_patch_g = get_saved_patches(f"{path}/channel1_new", len(patches))
    all_patch_b = get_saved_patches(f"{path}/channel2_new", len(patches))
    
    final_r = gaussian_combine_patches(all_patch_r, patch_info, img_size, overlap)
    final_g = gaussian_combine_patches(all_patch_g, patch_info, img_size, overlap)
    final_b = gaussian_combine_patches(all_patch_b, patch_info, img_size, overlap)
    
    return combine_to_rgb(final_r, final_g, final_b)

def process_channel(patches, channel, shift, save_data = False):
    """_summary_

    Args:
        patches (_type_): _description_
        channel (_type_): _description_
        save_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    deconvolved_imgs = []
    w12_vals = []
    channel_path = f"/home/alext12/Desktop/Thesis/Sem2/Polarisation Weighting/channel{channel}/"
    for i, patch in enumerate(patches):
        print(f"Processing patch {i + 1} of {len(patches)} channel {channel}")
        print(f"Patch shape: {patch.shape}")
        # trim patch to patch size
        deconvolved_img, w12val = ImageRun.run_estimate_w1_w2_patch(patch, channel, shift)
        if save_data:
            cv2.imwrite(f"{channel_path}patch_{i}.png", deconvolved_img*255)
        deconvolved_imgs.append(deconvolved_img)
        w12_vals.append(w12val)
    
    if save_data:
        with open(f'{channel_path}w12vals.pkl', 'wb') as f:
            pickle.dump(w12_vals, f)
            
    return deconvolved_imgs, w12_vals

def split_image_into_patches(image, patch_size, overlap):
    """
    Splits an RGB image into patches with given patch size and pixel-based overlap.
    
    Args:
        image (numpy array): Input image of shape (H, W, C) where C is the number of channels (3 for RGB).
        patch_size (tuple): Patch size as (patch_height, patch_width).
        overlap (int): Overlap size in pixels.
    
    Returns:
        patches (list of numpy arrays): List of image patches.
    """
    H, W, C = image.shape
    patch_height, patch_width = patch_size
    
    step_x = patch_width - overlap
    step_y = patch_height - overlap
    
    patches = []
    
    # Loop through the image and extract patches
    for y in range(0, H, step_y):
        for x in range(0, W, step_x):
            # Check if the patch exceeds image boundaries
            if y + patch_height <= H and x + patch_width <= W:
                patch = image[y:y + patch_height, x:x + patch_width, :]
                patches.append(patch)
    
    return patches

def reconstruct_image_from_patches(patches, image_shape, patch_size, overlap):
    """
    Reconstructs a single-channel image from patches by blending them together.

    Args:
        patches (list of numpy arrays): List of patches to be blended.
        image_shape (tuple): Shape of the original image (H, W).
        patch_size (tuple): Patch size as (patch_height, patch_width).
        overlap (int): Overlap size in pixels.
    
    Returns:
        reconstructed_image (numpy array): Reconstructed image of shape (H, W).
    """
    H, W = image_shape
    patch_height, patch_width = patch_size
    step_x = patch_width - overlap
    step_y = patch_height - overlap
    
    # Create an empty array for the reconstructed image and the weight mask
    reconstructed_image = np.zeros((H, W))
    weight_image = np.zeros((H, W))  # To accumulate weights for blending
    
    patch_index = 0
    for y in range(0, H, step_y):
        for x in range(0, W, step_x):
            if patch_index >= len(patches):
                break  # In case we run out of patches
            
            patch = patches[patch_index]
            patch_index += 1
            
            # Check if the patch can be placed within image bounds
            if y + patch_height <= H and x + patch_width <= W:
                # Create a blending mask to ensure smooth transitions between patches
                blending_mask = np.ones((patch_height, patch_width))
                blending_mask[:overlap, :] *= np.linspace(0, 1, overlap)[:, np.newaxis]  # Fade in the y-axis
                blending_mask[:, :overlap] *= np.linspace(0, 1, overlap)[np.newaxis, :]  # Fade in the x-axis
                
                # Add the patch and the blending mask to the corresponding location
                reconstructed_image[y:y + patch_height, x:x + patch_width] += patch * blending_mask
                weight_image[y:y + patch_height, x:x + patch_width] += blending_mask
    
    # Avoid division by zero
    weight_image[weight_image == 0] = 1
    
    # Normalize by the weight image to blend the patches
    reconstructed_image /= weight_image
    
    return reconstructed_image

def process_all_chanels(blurred_img, PATCH_SIZE):
    
    RED_CHANNEL = 0
    GREEN_CHANNEL = 1
    BLUE_CHANNEL = 2
    
    # shift_estimation = ShiftEstimate.compute_pixel_shift(blurred_img)
    shift_estimation = 5
    print(f"Shift estimate: {shift_estimation}")

    patches = PatchGetAndCombine.extract_image_patches_no_overlap(blurred_img, (PATCH_SIZE, PATCH_SIZE), shift_estimation)
    # patches = split_image_into_patches(blurred_img, (PATCH_SIZE, PATCH_SIZE), shift_estimation)
    # for i in range(len(patches)):
    #     # Scale the image from 0-1 to 0-255, then convert to uint8
    #     patch_rgb = cv2.cvtColor((patches[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    #     plt.figure()
    #     plt.imshow(patch_rgb)
    #     plt.axis('off')
    #     plt.show()

    
    deconvolved_imgs_r, w12_vals_r = process_channel(patches, RED_CHANNEL, shift_estimation, save_data=False)
    # reconstructed = reconstruct_image_from_patches(deconvolved_imgs_r, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), shift_estimation)
    _ = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap_with_quiver(deconvolved_imgs_r, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), w12_vals_r, 0, shift_estimation)
    deconvolved_imgs_g, w12_vals_g = process_channel(patches, GREEN_CHANNEL, shift_estimation, save_data=False)
    _ = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap_with_quiver(deconvolved_imgs_g, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), w12_vals_g, 1, shift_estimation)
    deconvolved_imgs_b, w12_vals_b = process_channel(patches, BLUE_CHANNEL, shift_estimation, save_data=False) 
    _ = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap_with_quiver(deconvolved_imgs_b, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), w12_vals_b, 2, shift_estimation)

    combined_r = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap(deconvolved_imgs_r, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), shift_estimation)
    combined_g = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap(deconvolved_imgs_g, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), shift_estimation)
    combined_b = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap(deconvolved_imgs_b, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), shift_estimation)

    combined_rgb = combine_to_rgb(combined_r, combined_g, combined_b)
    combined_w12 = np.mean([w12_vals_r, w12_vals_g, w12_vals_b], axis=0)
    return combined_rgb, combined_r, combined_g, combined_b, combined_w12

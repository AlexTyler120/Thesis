import PatchGetAndCombine
import ImageRun
import ShiftEstimate
import cv2
import pickle
import matplotlib.pyplot as plt

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
        deconvolved_img, w12val = ImageRun.run_estimate_w1_w2_patch(patch, channel, shift)
        if save_data:
            cv2.imwrite(f"{channel_path}patch_{i}.png", deconvolved_img*255)
        deconvolved_imgs.append(deconvolved_img)
        w12_vals.append(w12val)
    
    if save_data:
        with open(f'{channel_path}w12vals.pkl', 'wb') as f:
            pickle.dump(w12_vals, f)
            
    return deconvolved_imgs, w12_vals

def process_all_chanels(blurred_img, PATCH_SIZE):
    
    RED_CHANNEL = 0
    GREEN_CHANNEL = 1
    BLUE_CHANNEL = 2
    
    shift_estimation = ShiftEstimate.compute_pixel_shift(blurred_img)
    print(f"Shift estimate: {shift_estimation}")
    patches, patch_info, overlap = get_patches(blurred_img, PATCH_SIZE)
    
    deconvolved_imgs_r, w12_vals_r = process_channel(patches, RED_CHANNEL, shift_estimation, save_data=False)
    deconvolved_imgs_g, w12_vals_g = process_channel(patches, GREEN_CHANNEL, shift_estimation, save_data=False)
    deconvolved_imgs_b, w12_vals_b = process_channel(patches, BLUE_CHANNEL, shift_estimation, save_data=False) 
    overlap = (overlap[0]*3, overlap[1]*3)
    combined_r = PatchGetAndCombine.gaussian_blend_patches(deconvolved_imgs_r, patch_info, blurred_img.shape[:2], overlap)
    plt.figure()
    plt.imshow(combined_r, cmap='gray')
    plt.show()
    combined_g = PatchGetAndCombine.gaussian_blend_patches(deconvolved_imgs_g, patch_info, blurred_img.shape[:2], overlap)
    combined_b = PatchGetAndCombine.gaussian_blend_patches(deconvolved_imgs_b, patch_info, blurred_img.shape[:2], overlap)
    combined_rgb = combine_to_rgb(combined_r, combined_g, combined_b)
    
    return combined_rgb, combined_r, combined_g, combined_b
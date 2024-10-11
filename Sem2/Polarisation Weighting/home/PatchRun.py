import PatchGetAndCombine
import ImageRun
import ShiftEstimate
import Autocorrelation
import cv2
import pickle
import matplotlib.pyplot as plt
import Viewer
import numpy as np
import patchify



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


def process_channel(patches, channel, shift, save_data = False, original_patches = None):
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
    # for i, patch in enumerate(patches[470:], start=471):
    for i, patch in enumerate(patches):
        print(f"Processing patch {i + 1} of {len(patches)} channel {channel}")
        print(f"Patch shape: {patch.shape}")
        # trim patch to patch size
        if original_patches == None:
            deconvolved_img, w12val = ImageRun.run_estimate_w1_w2_patch(patch, channel, shift)
        else:
            deconvolved_img, w12val = ImageRun.run_estimate_w1_w2_patch(patch, channel, shift, og_patch = original_patches[i])
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
    shift_estimation = 5
    # # print(f"Shift estimate: {shift_estimation}")
    
    # patches = PatchGetAndCombine.extract_image_patch_overlap(blurred_img[:,:], (PATCH_SIZE, PATCH_SIZE))

    patches = PatchGetAndCombine.extract_image_patches_no_overlap(blurred_img, (PATCH_SIZE, PATCH_SIZE), shift_estimation)
    original_image = cv2.imread("python/test_im/fakefruit/small_fakefruit_0.png")[:, :, 2]
    patches_original = PatchGetAndCombine.extract_image_patches_no_overlap(original_image, (PATCH_SIZE, PATCH_SIZE), shift_estimation)
    deconvolved_imgs_r, w12_vals_r = process_channel(patches, RED_CHANNEL, shift_estimation, save_data=False, original_patches = None)
    # # deconvolved_imgs_g, w12_vals_g = process_channel(patches, GREEN_CHANNEL, shift_estimation, save_data=False)
    # # deconvolved_imgs_b, w12_vals_b = process_channel(patches, BLUE_CHANNEL, shift_estimation, save_data=False) 
    # save pickle
    # with open('channel0/rect_fruit_wi45.pkl', 'wb') as f:
    #     pickle.dump(deconvolved_imgs_r, f)
    # with open('channel0/rect_fruit_wi45__w12.pkl', 'wb') as f:
    #     pickle.dump(w12_vals_r, f)
    
    # with open('channel0/rect_fruit_rich.pkl', 'wb') as f:
    #     pickle.dump(deconvolved_imgs_r, f)
    # with open ('channel0/rect_fruit_rich__w12.pkl', 'wb') as f:
    #     pickle.dump(w12_vals_r, f)
    # read in pickle
    # with open('channel0/rect_fruit_rich.pkl', 'rb') as f:
    #     deconvolved_imgs_r = pickle.load(f)
    # with open ('channel0/rect_fruit_rich__w12.pkl', 'rb') as f:
    #     w12_vals_r = pickle.load(f)
    
    # with open('channel0/rect_fruit_wi.pkl', 'rb') as f:
    #     deconvolved_imgs_r = pickle.load(f)
    # with open('channel0/rect_fruit_wi__w12.pkl', 'rb') as f:
    #     w12_vals_r = pickle.load(f)
    _ = PatchGetAndCombine.reconstruct_image_patch_intensity(patches, deconvolved_imgs_r, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), shift_estimation, 0, w12_vals_r)
    # _ = PatchGetAndCombine.reconstruct_image_patch_intensity(patches, deconvolved_imgs_g, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), shift_estimation, 1, w12_vals_g)
    # _ = PatchGetAndCombine.reconstruct_image_patch_intensity(patches, deconvolved_imgs_b, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), shift_estimation, 2, w12_vals_g)
    # _ = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap_with_quiver(deconvolved_imgs_r, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), w12_vals_r, 0, shift_estimation)
    # _ = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap_with_quiver(deconvolved_imgs_g, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), w12_vals_g, 1, shift_estimation)
    # _ = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap_with_quiver(deconvolved_imgs_b, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), w12_vals_b, 2, shift_estimation)
    # combined_r = PatchGetAndCombine.reconstruct_image_from_patches_overlap(deconvolved_imgs_r, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE))
    # combined_g = PatchGetAndCombine.reconstruct_image_from_patches_overlap(deconvolved_imgs_g, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE))
    # combined_b = PatchGetAndCombine.reconstruct_image_from_patches_overlap(deconvolved_imgs_b, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE))
    # combined_r = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap(deconvolved_imgs_r, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), shift_estimation)
    # combined_g = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap(deconvolved_imgs_g, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), shift_estimation)
    # combined_b = PatchGetAndCombine.reconstruct_image_from_patches_no_overlap(deconvolved_imgs_b, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), shift_estimation)
    plt.show()
    combined_rgb = combine_to_rgb(combined_r, combined_g, combined_b)
    combined_w12 = np.mean([w12_vals_r, w12_vals_g, w12_vals_b], axis=0)
    return combined_rgb, combined_r, combined_g, combined_b, combined_w12

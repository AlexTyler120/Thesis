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


import numpy.ma as ma

def combine_to_rgb(final0, final1, final2):
    """Combine into RGB. Back to front because of BGR

    Args:
        final0 (_type_): _description_
        final1 (_type_): _description_
        final2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return cv2.merge([final0, final1, final2])


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
    # for i, patch in enumerate(patches[22:], start=22):
    for i, patch in enumerate(patches):
        print(f"Processing patch {i + 1} of {len(patches)} channel {channel}")
        print(f"Patch shape: {patch.shape}")

        # trim patch to patch size
        if original_patches == None:
            deconvolved_img, w12val, other_dec = ImageRun.run_estimate_w1_w2_patch(patch, channel, shift)
        else:
            deconvolved_img, w12val, other_dec = ImageRun.run_estimate_w1_w2_patch(patch, channel, shift, og_patch = original_patches[i])
        if save_data:
            cv2.imwrite(f"{channel_path}patch_{i}.png", deconvolved_img*255)
        
        deconvolved_imgs.append(deconvolved_img)
        w12_vals.append(w12val)
    
    if save_data:
        with open(f'{channel_path}w12vals.pkl', 'wb') as f:
            pickle.dump(w12_vals, f)
            
    return deconvolved_imgs, w12_vals, other_dec


# def process_channel(patches, channel, shift, save_data=False, original_patches=None):
#     """
#     Process each patch independently while averaging only immediate neighbor weights.
#     """
#     deconvolved_imgs = []
#     w12_vals = []
#     patch_map = {}  # Store `w1`, `w2` for each patch in its grid position (row, col)

#     num_rows = int(np.sqrt(len(patches)))  # Assuming patches are a grid; adjust as needed
#     num_cols = num_rows  # For simplicity, assuming square patch grid

#     for idx, patch in enumerate(patches):
#         print(f"Processing patch {idx + 1} of {len(patches)} channel {channel}")
#         row, col = divmod(idx, num_cols)
#         neighbor_weights = None

#         # Get neighbor weights only from immediate patches: left, above, or diagonal
#         neighbors = [(row - 1, col), (row, col - 1), (row - 1, col - 1)]
#         neighbor_w1, neighbor_w2, count = 0, 0, 0

#         for n_row, n_col in neighbors:
#             if (n_row, n_col) in patch_map:
#                 n_w1, n_w2 = patch_map[(n_row, n_col)]
#                 neighbor_w1 += n_w1
#                 neighbor_w2 += n_w2
#                 count += 1

#         if count > 0:
#             neighbor_weights = (neighbor_w1 / count, neighbor_w2 / count)

#         # Run the deconvolution with localized neighbor weights
#         if original_patches is None:
#             deconvolved_img, w12val, other_dec = ImageRun.run_estimate_w1_w2_patch(
#                 patch, channel, shift)#, neighbor_weights=neighbor_weights)
#         else:
#             deconvolved_img, w12val, other_dec = ImageRun.run_estimate_w1_w2_patch(
#                 patch, channel, shift, og_patch=original_patches[idx])#, neighbor_weights=neighbor_weights)

#         # Save the estimated `w1`, `w2` values for the current patch
#         patch_map[(row, col)] = w12val
#         deconvolved_imgs.append(deconvolved_img)
#         w12_vals.append(w12val)

#         # Optionally save data
#         if save_data:
#             cv2.imwrite(f"/path/to/save/channel_{channel}_patch_{idx}.png", deconvolved_img * 255)
    
#     return deconvolved_imgs, w12_vals, other_dec

def process_color(patches, shift, original_patches = None):
    deconvolved_imgs = []
    w12_vals = []
    for i, patch in enumerate(patches):
        print(f"Processing patch {i + 1} of {len(patches)}")
        print(f"Patch shape: {patch.shape}")
        # trim patch to patch size
        deconvolved_img, w12val = ImageRun.run_estimate_w1_w2_colour(patch, shift,  og_patch = original_patches[i])
        deconvolved_imgs.append(deconvolved_img)
        w12_vals.append(w12val)
        
    return deconvolved_imgs, w12_vals

def process_all_chanels(blurred_img, PATCH_SIZE, prefix, name):
    
    RED_CHANNEL = 0
    GREEN_CHANNEL = 1
    BLUE_CHANNEL = 2
    
    # shift_estimation = ShiftEstimate.compute_pixel_shift(blurred_img)
    shift_estimation = 5
    # # print(f"Shift estimate: {shift_estimation}")
    
    # patches = PatchGetAndCombine.extract_image_patches_no_overlap(blurred_img, (PATCH_SIZE, PATCH_SIZE), shift_estimation)
    # blurred_img = cv2.normalize(blurred_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    patches = PatchGetAndCombine.extract_image_patches_overlap(blurred_img, (PATCH_SIZE, PATCH_SIZE))
    
    original_image = cv2.imread("python/test_im/fakefruit/sq_fakefruit_90.png")
    # normalise original each cahnel;
    
    original_image = cv2.normalize(original_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    patches_original = PatchGetAndCombine.extract_image_patches_overlap(original_image, (PATCH_SIZE, PATCH_SIZE))
    filename = "wiener_0_12"
    # filename = "richard_2_12"
    deconvolved_imgs_r, w12_vals_r, _ = process_channel(patches, RED_CHANNEL, shift_estimation, save_data=False, original_patches = None)
    with open ("pickles/"+filename+"/"+prefix+name+"_red.pkl", 'wb') as f:
        pickle.dump(deconvolved_imgs_r, f)
    with open ("pickles/"+filename+"/"+prefix+name+"_w12_red.pkl", 'wb') as f:
      pickle.dump(w12_vals_r, f)    
    
    with open("pickles/"+filename+"/"+prefix+name+"_red.pkl", 'rb') as f:
        deconvolved_imgs_r = pickle.load(f)
    with open("pickles/"+filename+"/"+prefix+name+"_w12_red.pkl", 'rb') as f:
        w12_vals_r = pickle.load(f)

    im_r, angle_r, mag_r, _, _ = PatchGetAndCombine.reconstruct_image_patch_intensity_overlap(patches, deconvolved_imgs_r, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), 0, w12_vals_r)
    
    deconvolved_imgs_g, w12_vals_g, _ = process_channel(patches, GREEN_CHANNEL, shift_estimation, save_data=False, original_patches = None)
    with open ("pickles/"+filename+"/"+prefix+name+"_green.pkl", 'wb') as f:
        pickle.dump(deconvolved_imgs_g, f)
    with open ("pickles/"+filename+"/"+prefix+name+"_w12_green.pkl", 'wb') as f:
        pickle.dump(w12_vals_g, f)
    
    with open("pickles/"+filename+"/"+prefix+name+"_green.pkl", 'rb') as f:
        deconvolved_imgs_g = pickle.load(f)
    with open("pickles/"+filename+"/"+prefix+name+"_w12_green.pkl", 'rb') as f:
        w12_vals_g = pickle.load(f)
    im_g, angle_g, mag_g, _, _ = PatchGetAndCombine.reconstruct_image_patch_intensity_overlap(patches, deconvolved_imgs_g, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), 1, w12_vals_g)
    
    
    deconvolved_imgs_b, w12_vals_b, _ = process_channel(patches, BLUE_CHANNEL, shift_estimation, save_data=False, original_patches = None) 
    with open ("pickles/"+filename+"/"+prefix+name+"_blue.pkl", 'wb') as f:
        pickle.dump(deconvolved_imgs_b, f)
    with open ("pickles/"+filename+"/"+prefix+name+"_w12_blue.pkl", 'wb') as f:
        pickle.dump(w12_vals_b, f)

    
    with open("pickles/"+filename+"/"+prefix+name+"_blue.pkl", 'rb') as f:
        deconvolved_imgs_b = pickle.load(f)
    with open("pickles/"+filename+"/"+prefix+name+"_w12_blue.pkl", 'rb') as f:
        w12_vals_b = pickle.load(f)
    
    im_b, angle_b, mag_b, cd, ca = PatchGetAndCombine.reconstruct_image_patch_intensity_overlap(patches, deconvolved_imgs_b, blurred_img.shape[:2], (PATCH_SIZE, PATCH_SIZE), 2, w12_vals_b)
    
    reconstructed_image = combine_to_rgb(im_r, im_g, im_b)
    
    # average of angle
    # angle = ((angle_r*np.mean(im_r)) + (angle_g*np.mean(im_g)) + (angle_b*np.mean(im_b)))
    angle = (angle_r + angle_g + angle_b) / 3
    # angle = np.clip(angle, 25, 36)
    # average of magnitude
    mag = (mag_r + mag_g + mag_b) / 3
    plt.figure(figsize=(10, 10))
    plt.subplot(2,2,1)
    plt.imshow(reconstructed_image)
    plt.title("Reconstructed Image")
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.imshow(angle, cmap='jet')
    plt.title("Angle")
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(2,2,3)
    plt.imshow(mag, cmap='jet')
    plt.title("Magnitude")
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(2,2,4)
    plt.imshow(blurred_img)
    plt.title("Blurred Image")
    plt.axis('off')
    plt.figure()
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image)
    plt.axis('off')
    
    threshold = 0.2
    print(np.max(cd))

    mask_ang = ma.masked_where(cd < threshold, angle)
    mask_gt = ma.masked_where(cd < threshold, ca)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(mask_ang, cmap='jet')
    plt.title("Angle")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(mask_gt, cmap='jet')
    plt.title("Ground Truth")
    plt.axis('off')
    plt.colorbar()
    
    plt.show()

    # return

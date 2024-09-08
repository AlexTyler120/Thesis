import ImageRun
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PatchEstimate
import pickle

def plot_quiver_on_patches(image, patch_info, psf_vectors, patch_size, overlap):
    """
    Overlay a quiver plot on the patched image, representing the PSF for each patch.
    
    Parameters:
    image: The original image to display the quiver plot on.
    patch_info: List of (x0, y0, x1, y1) coordinates for each patch.
    psf_vectors: List of 1D PSF vectors (e.g., [dx, dy] for each patch).
    patch_size: Size of each patch (height, width).
    overlap: Overlap between patches.
    """
    fig, ax = plt.subplots()
    
    # Display the original image
    ax.imshow(image, cmap='gray')
    
    # Plot the quiver arrows
    for i, (x0, y0, x1, y1) in enumerate(patch_info):
        # Get the center of the patch
        patch_center_x = (x0 + x1) // 2
        patch_center_y = (y0 + y1) // 2
        
        # Extract the PSF vector for this patch (e.g., [dx, dy])
        psf_vector = psf_vectors[i][0]
        
        # Quiver: starts at (patch_center_x, patch_center_y) and moves by (dx, dy)
        ax.quiver(patch_center_x, patch_center_y, psf_vector[0], psf_vector[1], angles='xy', scale_units='xy', scale=0.1, color='red')
    
    plt.title("Quiver Plot Representing PSF Over Patches")
    # plt.show()

def main():
    RESIZE_VAR = 0.4
    GREY = False
    SIMULATED_SHIFT = 5
    WEIGHTING_SIM = 0.7
    ANGLE = 0

    ### Shift estimates with polarised images ###
    transformed_image = ImageRun.polarised_generation("ball", ANGLE, RESIZE_VAR, GREY, SIMULATED_SHIFT)
    ### ###

    ### Shift estimates with simulated images ###
    # transformed_image = ImageRun.simulated_generation("small_fakefruit_0.png", SIMULATED_SHIFT, RESIZE_VAR, GREY, WEIGHTING_SIM)
    ### ###

    ### Run estimation only getting w1 ###
    # ImageRun.run_estimate_w1(transformed_image)
    ### ###

    ### Run estimation getting w1 and w2 ###
    # ImageRun.run_estimate_w1_w2(transformed_image)
    ### ###
    patch_size = 25
    patches, patch_info, overlap = PatchEstimate.seperate_imgs_into_patches(transformed_image, patch_size)
    print(f"Number of patches: {len(patches)}")
    print(f"Patch overlap: {overlap}")
    all_patches = []
    overlap = (overlap[0]*2, overlap[1]*2)
    for i in range(len(patches)):
        img = cv2.imread(f"channel0_new/patch_{i}.png", cv2.IMREAD_GRAYSCALE)
        all_patches.append(img)
    final0 = PatchEstimate.gaussian_blend_patches(all_patches, patch_info, transformed_image.shape[:2], overlap)
    all_patches_1 = []
    
    for i in range(len(patches)):
        img = cv2.imread(f"channel1_new/patch_{i}.png", cv2.IMREAD_GRAYSCALE)
        all_patches_1.append(img)
    final1 = PatchEstimate.gaussian_blend_patches(all_patches_1, patch_info, transformed_image.shape[:2], overlap)
    all_patches_2 = []
    for i in range(len(patches)):
        img = cv2.imread(f"channel2_new/patch_{i}.png", cv2.IMREAD_GRAYSCALE)
        all_patches_2.append(img)
    final2 = PatchEstimate.gaussian_blend_patches(all_patches_2, patch_info, transformed_image.shape[:2], overlap)
    img_combined_rgb = cv2.merge([final2, final1, final0])
    plt.figure()
    plt.imshow(img_combined_rgb)
    plt.show()
    
    
    
    
    # deconvolved_imgs = []
    # middle_patch_index = len(patches) // 2
    # w12vals = []
    # channel0_path = "/home/alext12/Desktop/Thesis/Sem2/Polarisation Weighting/channel0_new/"
    # for i, patch in enumerate(patches):
    #     print(f"Processing patch {i + 1} of {len(patches)} channel 0")
    #     deconvolved_img, w12val = ImageRun.run_estimate_w1_w2_patch(patch, 0)
    #     # save image
    #     #convert deconvolved_img to 0-255
    #     cv2.imwrite(f"{channel0_path}patch_{i}.png", deconvolved_img*255)

        
    #     deconvolved_imgs.append(deconvolved_img)
    #     w12vals.append(w12val)
    # # Plot the quiver plot over the entire image
    # img_combined = PatchEstimate.combine_patches_into_image_with_overlap(deconvolved_imgs, patch_info, transformed_image.shape)
    # plt.figure()
    # plt.imshow(img_combined, cmap='gray')
    
    # # write w12 to pikle file
    # with open('{channel0_path}w12vals.pkl', 'wb') as f:
    #     pickle.dump(w12vals, f)
    
    # deconvolved_imgs1 = []
    # w12vals1 = []
    # channel1_path = "/home/alext12/Desktop/Thesis/Sem2/Polarisation Weighting/channel1_new/"
    # for i, patch in enumerate(patches):
    #     print(f"Processing patch {i + 1} of {len(patches)} channel 1")
    #     deconvolved_img, w12val = ImageRun.run_estimate_w1_w2_patch(patch, 1)
        
    #     cv2.imwrite(f"{channel1_path}patch_{i}.png", deconvolved_img*255)
        
    #     deconvolved_imgs1.append(deconvolved_img)
    #     w12vals1.append(w12val)
    # # Plot the quiver plot over the entire image
    # img_combined1 = PatchEstimate.combine_patches_into_image_with_overlap(deconvolved_imgs1, patch_info, transformed_image.shape)
    # plt.figure()
    # plt.imshow(img_combined1, cmap='gray')
    # with open('{channel1_path}w12vals.pkl', 'wb') as f:
    #     pickle.dump(w12vals1, f)
        
    # deconvolved_imgs2 = []
    # w12vals2 = []
    # channel2_path = "/home/alext12/Desktop/Thesis/Sem2/Polarisation Weighting/channel2_new/"
    # for i, patch in enumerate(patches):
    #     print(f"Processing patch {i + 1} of {len(patches)} channel 2")
    #     deconvolved_img, w12val = ImageRun.run_estimate_w1_w2_patch(patch, 2)
        
    #     cv2.imwrite(f"{channel2_path}patch_{i}.png", deconvolved_img*255)

    #     deconvolved_imgs2.append(deconvolved_img)
    #     w12vals2.append(w12val)
    # # Plot the quiver plot over the entire image
    # img_combined2 = PatchEstimate.combine_patches_into_image_with_overlap(deconvolved_imgs2, patch_info, transformed_image.shape)
    
    # plt.figure()
    # plt.imshow(img_combined2, cmap='gray')
    
    # with open('{channel2_path}w12vals.pkl', 'wb') as f:
    #     pickle.dump(w12vals2, f)
        
    # # merge all channels and plot rgb
    # img_combined_rgb = cv2.merge([img_combined2, img_combined1, img_combined])
    # plt.figure()
    # plt.imshow(img_combined_rgb)
    # plt.show()
if __name__ == "__main__":
    main()
import ImageRun
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PatchEstimate

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
    plt.show()

def main():
    RESIZE_VAR = 1
    GREY = False
    SIMULATED_SHIFT = 5
    WEIGHTING_SIM = 0.7
    ANGLE = 0

    ### Shift estimates with polarised images ###
    transformed_image = ImageRun.polarised_generation("fakefruit", ANGLE, RESIZE_VAR, GREY, SIMULATED_SHIFT)
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
    patch_size = 150

    patches, patch_info, overlap = PatchEstimate.seperate_imgs_into_patches(transformed_image, patch_size)
    print(f"Number of patches: {len(patches)}")

    deconvolved_imgs = []
    middle_patch_index = len(patches) // 2
    w12vals = []

    for i, patch in enumerate(patches):
        print(f"Processing patch {i + 1} of {len(patches)}")
        deconvolved_img, w12val = ImageRun.run_estimate_w1_w2(patch)
        deconvolved_imgs.append(deconvolved_img)
        w12vals.append(w12val)
    print(w12vals)
    # Plot the quiver plot over the entire image
    img_combined = PatchEstimate.combine_patches_into_image_with_overlap(deconvolved_imgs, patch_info, transformed_image.shape)

    plt.figure()
    plt.imshow(img_combined)
    plt.show()

    # Plot the quiver plot over the patches
    plot_quiver_on_patches(img_combined, patch_info, w12vals, patch_size, overlap)


if __name__ == "__main__":
    main()
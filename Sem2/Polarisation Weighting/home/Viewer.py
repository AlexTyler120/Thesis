import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def display_image(img, title, grey=False, showim=False):
    """
    Display an image with a given title.
    img: The image to display.
    title: The title of the image.
    grey: Whether the image is in grayscale.
    showim: Whether to display the image.
    """
    plt.figure(figsize=(10, 10))
    if grey:
        plt.imshow(img, cmap='gray')
    else:
        # bgr to rgb
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # img = cv2.convertScaleAbs(img, alpha=1.2, beta=5)
        plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    if showim:
        plt.show()
        
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
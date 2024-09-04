import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_image(image_path):
    """Loads an image from a file path."""
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def compute_difference(image1, image2):
    """Computes the absolute difference between two images."""
    difference = cv2.absdiff(image1, image2)
    
    # Convert the difference to grayscale if desired (optional step)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    
    # Normalize the grayscale difference image to the range [0, 1]
    norm_difference = cv2.normalize(gray_difference, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return norm_difference

def display_images_and_heatmap(image1, image2, difference):
    """Displays the original images side by side and the normalized difference as a heatmap with a color bar."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convert BGR images to RGB for display
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    ax[0].imshow(image1_rgb)
    ax[0].set_title('Image 1')
    ax[0].axis('off')

    ax[1].imshow(image2_rgb)
    ax[1].set_title('Image 2')
    ax[1].axis('off')

    plt.figure(figsize=(6, 5))

    # Display the normalized difference as a heatmap
    heatmap = plt.imshow(difference, cmap='jet')
    plt.title('Difference Heatmap (Normalized)')
    plt.axis('off')

    # Add a color bar to the heatmap
    cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    cbar.set_label('Difference Intensity')

    plt.show()

def main(image_path1, image_path2):
    """Main function to load images, compute difference, and display results."""
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)

    # Ensure images are of the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must be the same size for comparison")

    difference = compute_difference(image1, image2)
    display_images_and_heatmap(image1, image2, difference)

# Example usage
image_path1 = 'python/test_im/bottles/bottles_0.png'
image_path2 = 'python/test_im/bottles/bottles_90.png'
main(image_path1, image_path2)

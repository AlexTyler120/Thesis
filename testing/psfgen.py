import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def display_image_and_get_point(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title("Click on the center of the visible light source")
    point = []

    def onclick(event):
        x, y = int(event.xdata), int(event.ydata)
        point.append((y, x))  # Note the order: (row, column)
        ax.plot(x, y, 'ro')  # Mark the selected point
        fig.canvas.draw()
        plt.close(fig)  # Close the figure after the click

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return point[0]

def extract_psf(image, center, size=100):
    x, y = center
    half_size = size // 2
    psf = image[x-half_size:x+half_size, y-half_size:y+half_size]
    psf_normalized = psf / np.sum(psf)
    return psf_normalized

image_path = r"C:/Users/Alex/OneDrive - The University of Sydney (Students)/UNI/Thesis/testing/pointlight/0degpolarisation_dualpinwpol.bmp"
image = load_image(image_path)

# Get center interactively
center = display_image_and_get_point(image)

# Extract and normalize the PSF
psf = extract_psf(image, center, size=200)

# Display the PSF
plt.imshow(psf, cmap='gray')
plt.title("Extracted PSF")
plt.colorbar()
plt.show()

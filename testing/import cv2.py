import cv2
import numpy as np
from scipy.signal import convolve2d, wiener
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import correlate2d
from sklearn.decomposition import PCA, FastICA

def getLorikeetImage(object, distance):
    try:
        image_path = f'testing\pol\camera\Lights Off\{object}\{distance}cm_raw.bmp'
        image = cv2.imread(image_path)
        # image = cv2.rotate(image, cv2.ROTATE_180)
    except:
        image = None
    return image

def sharpen_image(image):
    sharpening_kernel = np.array([[-1,-1,-1], 
                                  [-1, 7,-1],
                                  [-1,-1,-1]])
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened_image

def removeNoise(image):
    # cv2.imshow('Original', image)
    median_filtered = cv2.medianBlur(image, 15)
    # cv2.imshow('Median Filtered', median_filtered)
    bilateral_filtered = cv2.bilateralFilter(median_filtered, 15, 100, 100)
    # cv2.imshow('Bilateral Filtered', bilateral_filtered)
    denoised_image = cv2.fastNlMeansDenoisingColored(bilateral_filtered, None, 15, 15, 7, 21)
    # cv2.imshow('Sharpened Image', sharpened_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return bilateral_filtered
    

def measureBlur(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def contrastEnhance(image):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2,a,b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced_image

def colourSat(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Increase saturation
    hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], 1.5)
    # Convert back to BGR
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return saturated_image

def disparity_comput(image1, image2):
    grey1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(grey1, grey2).astype(np.float32)/16.0
    return disparity

def removeExtra(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to create mask
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    # Find all connected components in the mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Find the label of the largest component (excluding the background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create a new mask where only the largest component is white
    new_mask = np.where(labels == largest_label, 255, 0).astype('uint8')
    image_no_black = cv2.bitwise_and(image, image, mask=new_mask)

    return image_no_black

# def main():
#     images = [60, 80]
#     for num in images:
#         image = getLorikeetImage('Lorikeet', num)
#         image = removeExtra(removeNoise(image))
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply edge detection
#     edges = cv2.Canny(gray_image, 10, 10)

#     # Display the original and edge-detected images
#     plt.figure(figsize=(15, 10))
#     plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122), plt.imshow(edges, cmap='gray')
#     plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])
#     plt.show()

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Create masks for each contour
#     mask1 = np.zeros_like(gray_image)
#     mask2 = np.zeros_like(gray_image)

#     for contour in contours:
#         # Approximate contour to a polygon
#         epsilon = 0.01 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
        
#         # Find bounding box and shift mask
#         x, y, w, h = cv2.boundingRect(approx)
        
#         # Depending on the position, decide which mask to use
#         if x < gray_image.shape[1] // 2:
#             cv2.drawContours(mask1, [contour], -1, 255, thickness=cv2.FILLED)
#         else:
#             cv2.drawContours(mask2, [contour], -1, 255, thickness=cv2.FILLED)

#     # Separate images using the masks
#     separated_image1 = cv2.bitwise_and(image, image, mask=mask1)
#     separated_image2 = cv2.bitwise_and(image, image, mask=mask2)

#     # Display the separated images
#     plt.figure(figsize=(15, 10))
#     plt.subplot(121), plt.imshow(cv2.cvtColor(separated_image1, cv2.COLOR_BGR2RGB))
#     plt.title('Separated Image 1'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122), plt.imshow(cv2.cvtColor(separated_image2, cv2.COLOR_BGR2RGB))
#     plt.title('Separated Image 2'), plt.xticks([]), plt.yticks([])
#     plt.show()
# if __name__ == '__main__':
#     main()
def main():
    cv2.setNumThreads(32)
    images = [60, 80]
    for num in images:
        image = getLorikeetImage('Lorikeet', num)
        image = removeNoise(image)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray_image, 10, 15)

    # Display the original and edge-detected images
    plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Calculate the phase correlation
    dft1 = cv2.dft(np.float32(edges), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(np.float32(edges), flags=cv2.DFT_COMPLEX_OUTPUT)
    cross_power_spectrum = (dft1 * dft2.conjugate()) / np.abs(dft1 * dft2.conjugate())
    cross_power_spectrum = np.fft.ifftshift(cross_power_spectrum)
    shift = np.fft.ifft2(cross_power_spectrum)
    max_loc = np.unravel_index(np.argmax(np.abs(shift)), shift.shape)
    shift_x = max_loc[1] - edges.shape[1] // 2

    print(f"Detected horizontal shift: {shift_x} pixels")

    # Align the images based on the shift
    shifted_image = np.roll(image, shift_x, axis=1)

    # Create masks based on the alignment
    mask1 = np.zeros_like(gray_image)
    mask2 = np.zeros_like(gray_image)

    if shift_x > 0:
        mask1[:, :-shift_x] = 255
        mask2[:, shift_x:] = 255
    else:
        mask1[:, -shift_x:] = 255
        mask2[:, :shift_x] = 255

    # Separate images using the masks
    separated_image1 = cv2.bitwise_and(image, image, mask=mask1)
    separated_image2 = cv2.bitwise_and(shifted_image, shifted_image, mask=mask2)

    # Display the separated images
    plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.imshow(cv2.cvtColor(separated_image1, cv2.COLOR_BGR2RGB))
    plt.title('Separated Image 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(separated_image2, cv2.COLOR_BGR2RGB))
    plt.title('Separated Image 2'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()
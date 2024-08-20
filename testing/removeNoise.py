import cv2
import numpy as np

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
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened_image

def removeNoise(image):
    # cv2.imshow('Original', image)
    median_filtered = cv2.medianBlur(image, 11)
    # cv2.imshow('Median Filtered', median_filtered)
    bilateral_filtered = cv2.bilateralFilter(median_filtered, 11, 75, 75)
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

def main():
    images = [60,80]
    for num in images:
        image = sharpen_image(removeNoise(getLorikeetImage('Lorikeet', num)))
        print(measureBlur(image))
        # contrast_image = colourSat((contrastEnhance(image)))
        cv2.imshow(f'{num}', image)
        # cv2.imwrite(f'testing\pol\camera\Lights Off\Lorikeet\Preprocessed\{num}cm_noise_rem.bmp', image)
        # cv2.imshow(f'{num} enhanced', contrast_image)s

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
import cv2
import Viewer

def read_image(path, size, grey):
    image = cv2.imread(path)
    if grey:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (0,0), fx=size, fy=size)
    height = image.shape[0]
    width = image.shape[1]

    return image, height, width

def create_shifted_image(img1, img2, shift, show=False):
    img2_shifted = img2.copy()
    # apply a shift to the image 
    img2_shifted[:, shift:] = img1[:, :-shift]
    # combine images
    IM_WEIGHT = 0.5
    image_transformed = cv2.addWeighted(img1, IM_WEIGHT, img2_shifted, IM_WEIGHT, 0)
    if show:
        Viewer.display_image(image_transformed, "Shifted Image", showim=show)
    return image_transformed

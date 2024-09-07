import cv2
import numpy as np
import Viewer
import matplotlib.pyplot as plt
def read_image(path, size, grey):
    image = cv2.imread(path)
    if grey:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    image = cv2.resize(image, (0,0), fx=size, fy=size)
    height = image.shape[0]
    width = image.shape[1]

    return image, height, width

def create_shifted_image_polarised_imgs(img1, img2, shift, show=False):
    img2_shifted = img2.copy()
    # apply a shift to the image 
    img2_shifted[:, shift:] = img2_shifted[:, :-shift]
    # combine images
    IM_WEIGHT = 0.5
    image_transformed = cv2.addWeighted(img1, IM_WEIGHT, img2_shifted, IM_WEIGHT, 0)
    if show:
        Viewer.display_image(image_transformed, "Shifted Image", showim=show)
    return image_transformed / np.max(image_transformed)

def create_shifted_simulation(img1, w1, shift):
    img2 = img1.copy()
    # apply a shift to the image 
    img2[:, shift:] = img1[:, :-shift]
    # combine images
    image_transformed = cv2.addWeighted(img1, w1, img2, 1-w1, 0)
    Viewer.display_image(image_transformed, "Shifted Image", showim=True)
    # image_transformed = img1*w1 + img2*(1-w1)
    return image_transformed / np.max(image_transformed)

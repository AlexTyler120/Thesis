import cv2
import numpy as np
import Viewer
import matplotlib.pyplot as plt
def read_image(path, size, grey):
    image = cv2.imread(path)
    if grey:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (0,0), fx=size, fy=size)
    height = image.shape[0]
    width = image.shape[1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.figure()
    # plt.imshow(image_rgb)
    # plt.show()

    return image, height, width

def create_shifted_image_polarised_imgs(img1, img2, shift, show=False):
    img2_shifted = img2.copy()
    # apply a shift to the image 
    img2_shifted[:, shift:] = img2_shifted[:, :-shift]
    # combine images
    IM_WEIGHT = 0.5
    # img1[:, :, 0] = cv2.equalizeHist(img1[:, :, 0])
    # # channel 1
    # img1[:, :, 1] = cv2.equalizeHist(img1[:, :, 1])
    # # channel 2
    # img1[:, :, 2] = cv2.equalizeHist(img1[:, :, 2])
    
    # img2_shifted[:, :, 0] = cv2.equalizeHist(img2_shifted[:, :, 0])
    # # channel 1
    # img2_shifted[:, :, 1] = cv2.equalizeHist(img2_shifted[:, :, 1])
    # # channel 2
    # img2_shifted[:, :, 2] = cv2.equalizeHist(img2_shifted[:, :, 2])
    image_transformed = cv2.addWeighted(img1, IM_WEIGHT, img2_shifted, IM_WEIGHT, 0)
    # histogram equalise each channel
    # channel 0 
    
    # image_transformed = cv2.cvtColor(image_transformed, cv2.COLOR_BGR2RGB)
    if show:
        Viewer.display_image(image_transformed, "Shifted Image", showim=show)
    # convert to rgb
    
    return image_transformed / np.max(image_transformed)

def create_shifted_image_polarised_y(img1, img2, shift, show=False):
    img2_shifted = img2.copy()
    # apply a shift to the image 
    img2_shifted[shift:, :] = img2_shifted[:-shift, :]
    # combine images
    IM_WEIGHT = 1
    image_transformed = cv2.addWeighted(img1, IM_WEIGHT, img2_shifted, IM_WEIGHT, 0)
    # rotate iamge 90 degrees
    image_transformed = cv2.rotate(image_transformed, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # histogram equalise each channel
    # channel 0 
    # image_transformed[:, :, 0] = cv2.equalizeHist(image_transformed[:, :, 0])
    # # channel 1
    # image_transformed[:, :, 1] = cv2.equalizeHist(image_transformed[:, :, 1])
    # # channel 2
    # image_transformed[:, :, 2] = cv2.equalizeHist(image_transformed[:, :, 2])
    if show:
        Viewer.display_image(image_transformed, "Shifted Image", showim=show)
        
    return image_transformed / np.max(image_transformed)

def create_shifted_simulation(img1, w1, shift):
    img2 = img1.copy()
    # apply a shift to the image 
    img2[:, shift:] = img1[:, :-shift]
    # combine images
    image_transformed = cv2.addWeighted(img1, w1, img2, 1-w1, 0)
    Viewer.display_image(image_transformed, "Shifted Image", showim=False)
    # image_transformed = img1*w1 + img2*(1-w1)
    return image_transformed / np.max(image_transformed)

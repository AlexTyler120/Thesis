import ImageRun
import PatchRun
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import PatchGetAndCombine
from matplotlib.colors import LogNorm
import WeightingEstimate
import Autocorrelation
from mpl_toolkits.mplot3d import Axes3D
from gt import gt

def main():
    RESIZE_VAR = 1
    GREY = False
    SIMULATED_SHIFT = 5
    WEIGHTING_SIM = 0.6
    ANGLE = 0
    PATCH_SIZE = 12
    item = "fakefruit"
    prefix = "rect_"
    ### Shift estimatrec1es with polarised images ###
    transformed_image = ImageRun.polarised_generation(item, prefix, ANGLE, RESIZE_VAR, GREY, SIMULATED_SHIFT)
    ### ###
    
    gt()
    rgb, r, g, b, w12 = PatchRun.process_all_chanels(transformed_image, PATCH_SIZE, prefix, item)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2,2,1)
    plt.imshow(r, cmap='gray')
    plt.title("Red Channel")
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.imshow(g, cmap='gray')
    plt.title("Green Channel")
    plt.axis('off')
    
    plt.subplot(2,2,3)
    plt.imshow(b, cmap='gray')
    plt.title("Blue Channel")
    plt.axis('off')
    
    plt.subplot(2,2,4)
    plt.imshow(rgb)
    plt.title("RGB Image")
    plt.axis('off')
    # _ = PatchGetAndCombine.create_full_quiver_with_overlap(rgb, transformed_image.shape[:2], (PATCH_SIZE, PATCH_SIZE), w12)
    plt.show()
    
    
if __name__ == "__main__":
    main()
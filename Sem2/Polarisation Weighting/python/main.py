import Image
import shiftImage
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import skimage as sk
import plotly.graph_objects as go
import plotly.subplots as spgraph
import cv2


def main():
    path = "python/nopol.jpg"
    size = 0.1
    grey = True

    original_image = Image.Image(path, size, grey)
    w1 = 0.3
    w2 = 0.7

    shift = 10

    shifted = shiftImage.shiftImage(original_image, w1, w2, shift)
    shifted.computePixelShift()

    w1guess = 0.5
    bounds = [(0, 1)]
    method = "Powell"
    shifted.opt_minimise_weights(w1guess, bounds, method)
    
if __name__ == "__main__":
    main()

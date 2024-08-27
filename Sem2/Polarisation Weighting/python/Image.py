import cv2
import matplotlib.pyplot as plt
import numpy as np

class Image:
    def __init__(self, path, size, grey):
        self.original_image = cv2.imread(path)
        if grey:
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.original_image = cv2.resize(self.original_image, (0,0), fx=size, fy=size)
        self.image_height = self.original_image.shape[0]
        self.image_width = self.original_image.shape[1]
        self.grey = grey
        print("imaege maede")
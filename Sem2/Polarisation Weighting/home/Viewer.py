import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def display_image(img, title, grey=False, showim=False):
    plt.figure(figsize=(10, 10))
    if grey:
        plt.imshow(img, cmap='gray')
    else:
        # bgr to rgb
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    if showim:
        plt.show()
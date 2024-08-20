from removeNoise import removeNoise
import cv2

image_name = 'allfoc'
image = cv2.imread(f'testing\pol\midlightint\{image_name}.bmp')

image = removeNoise(image)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
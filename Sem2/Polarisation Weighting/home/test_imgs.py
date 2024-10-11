import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

# pickel read
def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

I0 = read_pickle("I_0_total.pkl")
I45 = read_pickle("I_45_total.pkl")
I90 = read_pickle("I_90_total.pkl")
I135 = read_pickle("I_135_total.pkl")
# clip i135 0 -1
# I135 = np.clip(I135, 0, 1)
# normalise
# 
I0 = cv2.normalize(I0, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

I45 = cv2.normalize(I45, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
I0 = I45
I90 = cv2.normalize(I90, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

I135 = cv2.normalize(I135, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
I90 = I135
# blur images
# I0 = cv2.GaussianBlur(I0, (9, 9), 0)
# I45 = cv2.GaussianBlur(I45, (9, 9), 0)
# I90 = cv2.GaussianBlur(I90, (9, 9), 0)
# I135 = cv2.GaussianBlur(I135, (9, 9), 0)
# plt show all in one 
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(I0, cmap='jet')
plt.title("I0")
plt.colorbar()
plt.subplot(2, 2, 2)
plt.imshow(I45, cmap='jet')
plt.title("I45")
plt.subplot(2, 2, 3)
plt.imshow(I90, cmap='jet')
plt.colorbar()
plt.title("I90")
plt.subplot(2, 2, 4)
plt.imshow(I135, cmap='jet')
# plt.show()

# stokes params
S0 = 0.5 * (I0 + I45 + I90 + I135)
S1 = -I0 + I90
S2 = -I45 + I135

#dolp
dolp = np.sqrt(S1**2 + S2**2) / S0
#aolp
aolp = 0.5 * np.arctan(S2 / S1)

# plt show all in one
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(dolp, cmap='jet')
plt.title("DoLP")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(np.degrees(aolp), cmap='seismic')
plt.title("AoLP")
plt.colorbar()
plt.figure()
diff135 = np.abs(I135 - I90)
diff45 = np.abs(I45 - I0)
diff135[diff135 > 0.15] = 0.15
diff45[diff45 > 0.15] = 0.15
plt.imshow(diff45, cmap='jet')
plt.colorbar()
plt.title("45 - I0")
plt.figure()
plt.imshow(diff135, cmap='jet')
plt.colorbar()
plt.title("I135 - I90")
plt.figure()
# normalise i90/i0
I_900 = I90 - I0
I_900 = np.clip(I_900, -0.2, 1)
plt.imshow(I_900, cmap='jet')
plt.colorbar()
plt.title("I90/I0")


i0path = 'python/test_im/fakefruit/rect_fakefruit_0.png'
i90path = 'python/test_im/fakefruit/rect_fakefruit_90.png'
i135path = 'python/test_im/fakefruit/rect_fakefruit_135.png'
i45path = 'python/test_im/fakefruit/rect_fakefruit_45.png'
I0gt = cv2.imread(i0path)
I45gt = cv2.imread(i45path)
I90gt = cv2.imread(i90path)
I135gt = cv2.imread(i135path)
# Normalize each channel to range 0-1
I0gt = cv2.normalize(I0gt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
I45gt = cv2.normalize(I45gt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
I90gt = cv2.normalize(I90gt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
I135gt = cv2.normalize(I135gt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.abs(I135gt[:,:, 2] - I90gt[:,:, 2]), cmap='jet')
plt.colorbar()
plt.title("I135 - I90")
plt.subplot(1, 2, 2)
plt.imshow(np.abs(I45gt[:,:, 2] - I0gt[:,:, 2]), cmap='jet')
plt.colorbar()
plt.title("I45 - I0")
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(I0gt[:,:, 2], cmap='jet')
plt.title("I0")
plt.subplot(2, 2, 2)
plt.imshow(I45gt[:,:, 2], cmap='jet')
plt.title("I45")
plt.subplot(2, 2, 3)
plt.imshow(I90gt[:,:, 2], cmap='jet')
plt.title("I90")
plt.subplot(2, 2, 4)
plt.imshow(I135gt[:,:, 2], cmap='jet')
plt.title("I135")
plt.figure()
plt.imshow(I90gt[:,:, 2]-I0gt[:,:, 2], cmap='jet')
plt.title("I90/I0 ground truth")
plt.colorbar()
plt.show()
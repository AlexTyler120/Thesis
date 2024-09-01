import numpy as np
import cv2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def polarization_model(theta, I_unpolarized, I_polarized, phi):
    return I_unpolarized + I_polarized * np.cos(2 * (theta - phi))

def fit_pixel(pixel_intensities, angles):
    popt, _ = curve_fit(polarization_model, angles, pixel_intensities, p0=[np.mean(pixel_intensities), 1, 0])
    I_unpolarized, I_polarized, phi = popt
    AoP = phi
    DoP = I_polarized / (I_unpolarized + I_polarized)
    return AoP, DoP, I_unpolarized

def fit_polarization(images, angles):
    rows, cols = images[0].shape
    AoP = np.zeros((rows, cols))
    DoP = np.zeros((rows, cols))
    I_unpolarized = np.zeros((rows, cols))

    pixel_intensities_all = np.stack(images, axis=-1)
    
    # Parallel processing
    results = Parallel(n_jobs=24)(delayed(fit_pixel)(pixel_intensities_all[i, j, :], angles) 
                                  for i in range(rows) 
                                  for j in range(cols))
    
    for idx, (aop, dop, i_unpol) in enumerate(results):
        i = idx // cols
        j = idx % cols
        AoP[i, j] = aop
        DoP[i, j] = dop
        I_unpolarized[i, j] = i_unpol

    return AoP, DoP, I_unpolarized

def calculate_normals(AoP, DoP):
    theta_p = np.arccos(DoP)
    normals = np.zeros((AoP.shape[0], AoP.shape[1], 3))
    normals[:, :, 0] = np.cos(AoP) * np.sin(theta_p)
    normals[:, :, 1] = np.sin(AoP) * np.sin(theta_p)
    normals[:, :, 2] = np.cos(theta_p)

    # Normalize the normals
    norms = np.linalg.norm(normals, axis=2)
    normals[:, :, 0] /= norms
    normals[:, :, 1] /= norms
    normals[:, :, 2] /= norms

    return normals

# Assuming you have loaded your images into these variables
images = [
    cv2.imread('python/test_im/ball/ball_0.png', cv2.IMREAD_GRAYSCALE),
    cv2.imread('python/test_im/ball/ball_45.png', cv2.IMREAD_GRAYSCALE),
    cv2.imread('python/test_im/ball/ball_90.png', cv2.IMREAD_GRAYSCALE),
    cv2.imread('python/test_im/ball/ball_135.png', cv2.IMREAD_GRAYSCALE)
]

angles = np.array([0, 45, 90, 135]) * np.pi / 180  # Convert to radians

# Fit polarization model and calculate AoP, DoP
AoP, DoP, I_unpolarized = fit_polarization(images, angles)

# Calculate surface normals
normals = calculate_normals(AoP, DoP)

# Visualize the surface normals (as an RGB image)
normal_map = (normals + 1) / 2  # Scale to 0-1 for visualization
plt.imshow(normal_map)
plt.show()

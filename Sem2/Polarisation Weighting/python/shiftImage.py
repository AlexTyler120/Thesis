import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import skimage as sk

class shiftImage:
    def __init__(self, img, w1, w2, shift):
        self.grey = img.grey
        self.It = np.zeros(img.original_image.shape)
        self.image_height = img.image_height
        self.image_width = img.image_width

        # shift
        I1 = img.original_image.copy()
        I2 = I1.copy()

        I2[:, :-shift] = I2[:, shift:]
        if not self.grey:
            self.It = (w1 * I1.astype(int) + w2 * I2.astype(int)).astype(int)
        else:
            self.It = (w1 * I1 + w2 * I2)


        ## estimated vals
        self.estimated_shift = None
        self.estimated_psf = None

    def computeCrossCorrelation(self, img):
        # max_shift = self.image_width // 2
        max_shift = 20
        shift_vals = []
        corr_vals = []

        for x_shift in range(-max_shift, max_shift + 1):
            It_shifted = sp.ndimage.shift(img, shift=(0, x_shift), mode='constant', cval=0)

            # flatten
            It_flat = img.flatten()
            It_shifted_flat = It_shifted.flatten()

            cross_corr = np.correlate(It_flat, It_shifted_flat, mode="valid")

            corr_val = np.max(cross_corr)

            shift_vals.append(x_shift)
            corr_vals.append(corr_val)
        
        shift_vals = np.array(shift_vals)
        corr_vals = np.array(corr_vals)
        

        return shift_vals, corr_vals
    
    def applyFilter(self, corr_vals):
        windowLength = 5
        polyorder = 3
        baseline = sp.signal.savgol_filter(corr_vals, windowLength, polyorder)

        return corr_vals - baseline
    
    def obtainCorrelationPeaks(self, shift_vals, corr_vals):
        peaks = []
        steepness = []
        for i in range(len(shift_vals)):
            if i == 0 or i == len(shift_vals) - 1 or i == ((len(shift_vals)) - 1)/2:
                continue
            else:
                left_corr = corr_vals[i-1]
                right_corr = corr_vals[i+1]
                mid_corr = corr_vals[i]
                left_rise = mid_corr - left_corr
                right_rise = mid_corr - right_corr
                if left_rise > 0 and right_rise > 0:
                    peaks.append(i)
                    steepness.append(left_rise + right_rise)           
        return peaks, steepness
    
    def computePixelShift(self):
        shift_vals, corr_vals = self.computeCrossCorrelation(self.It)
        corr_vals = self.applyFilter(corr_vals)
        peaks, steepness = self.obtainCorrelationPeaks(shift_vals, corr_vals)
        
        # sort peaks
        sorted_peaks = [x for _, x in sorted(zip(steepness, peaks), reverse=True)]
        top_peaks = sorted_peaks[:2]
        peak_shifts = [shift_vals[peak] for peak in top_peaks]
        if abs(peak_shifts[0]) == abs(peak_shifts[1]):
            estiamted_shift = abs(peak_shifts[0])
        else:
            # raise error
            print("Error: Multiple peaks detected")
            estiamted_shift = None
        self.estimated_shift = estiamted_shift

    def getImagePSF(self, w1_est):

        psf = np.zeros(self.estimated_shift + 1)
        psf[-1] = w1_est
        psf[0] = 1 - w1_est

        psf = psf/np.sum(psf)

        psf = np.expand_dims(psf, axis=0)
        self.estimated_psf = psf
        print(psf)

    def testPSF(self, img):
        new_blur = sp.ndimage.convolve(img.original_image, self.estimated_psf, mode='constant')

        plt.figure(figsize=(12, 4))
        shifted_back_it = sp.ndimage.shift(self.It, shift=(0, self.estimated_shift//2), mode='constant', cval=0)
        plt.subplot(1, 3, 1)
        plt.imshow(shifted_back_it, cmap='gray')
        plt.title('Original Blurred Image')

        plt.subplot(1, 3, 2)
        plt.imshow(new_blur, cmap='gray')
        plt.title('New Blurred Image')

        # Calculate the difference between original_blurred and the new blurred image and shift back for comparison
        
        difference = np.abs(shifted_back_it.astype(float) - new_blur.astype(float))
        print(np.max(difference))

        plt.subplot(1, 3, 3)
        sns.heatmap(difference, cmap='hot', cbar=True, square=True)
        plt.title('Difference between Original blurred and Estimated blurred')

        plt.show()
        return new_blur
    
    def loss_func(self, est):
        w1_est = est

        img = self.It.copy()

        # Contrast enhance image
        img = img / np.max(img)

        img = sp.ndimage.shift(img, shift=(0, -self.estimated_shift//2), mode='constant', cval=0)

        # Compute PSF and perform Wiener deconvolution
        self.getImagePSF(w1_est)
        deconvolved = sk.restoration.wiener(img, self.estimated_psf, balance=0)

        # Use your computeCrossCorrelation function
        shift_vals, corr_vals = self.computeCrossCorrelation(deconvolved)

        # Find the index of the peak at zero shift
        zero_idx = np.where(shift_vals == 0)[0][0]

        loss = 0

        # Check for gradients on the left side (should be positive)
        for i in range(1, zero_idx):
            if corr_vals[i] <= corr_vals[i-1]:  # Gradient should be positive, but it's not
                loss += abs(corr_vals[i] - corr_vals[i-1])

        # Check for gradients on the right side (should be negative)
        for i in range(zero_idx + 1, len(corr_vals)):
            if corr_vals[i] >= corr_vals[i-1]:  # Gradient should be negative, but it's not
                loss += abs(corr_vals[i] - corr_vals[i-1])

        print(f"loss: {loss}")
        #print mse
        # print(f"mse: {np.mean((img - blur_rec)**2)*100}")
        return loss
    
    def opt_minimise_weights(self, w1guess, bounds, method):
        result = sp.optimize.minimize(self.loss_func, w1guess, bounds=bounds, method=method)
        return result.x
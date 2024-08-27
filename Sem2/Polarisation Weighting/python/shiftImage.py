import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import skimage as sk
import multiprocessing as mp
class EarlyStoppingException(Exception):
    pass

class shiftImage:
    def __init__(self, img, w1, w2, shift):
        self.grey = img.grey
        self.It = np.zeros(img.original_image.shape)
        self.image_height = img.image_height
        self.image_width = img.image_width
        # self.It = img.original_image.copy()
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
        self.min_loss = None
        self.best_w1 = None

    def computeCrossCorrelation(self, img, normalised=True):
        if self.estimated_shift is None:
            max_shift = self.image_width // 2
        else:
            max_shift = self.estimated_shift*2

        shift_vals = []
        corr_vals = []

        if normalised:
            img_mean = np.mean(img)
            img_std = np.std(img)
            img_norm = (img - img_mean) / img_std

        for x_shift in range(-max_shift, max_shift + 1):
            It_shifted = sp.ndimage.shift(img, shift=(0, x_shift), mode='constant', cval=0)

            if normalised:
                It_shifted_mean = np.mean(It_shifted)
                It_shifted_std = np.std(It_shifted)
                if It_shifted_std == 0:
                    It_shifted_std = 1
                It_shifted_normalised = (It_shifted - It_shifted_mean) / It_shifted_std
                It_flat = img_norm.flatten()
                It_shifted_flat = It_shifted_normalised.flatten()
                numerator = np.correlate(It_flat, It_shifted_flat, mode="valid")
                denominator = np.linalg.norm(It_flat) * np.linalg.norm(It_shifted_flat)
                cross_corr = numerator / denominator
            # flatten
            else:
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
        windowLength = 7
        polyorder = 3
        baseline = sp.signal.savgol_filter(corr_vals, windowLength, polyorder)
        return corr_vals - baseline
    
    def obtainCorrelationPeaks(self, shift_vals, corr_vals):
        peaks = []
        steepness = []
        for i in range(len(shift_vals) - 1):
            zero_idx = np.where(shift_vals == 0)[0][0]
            if i == 0 or i == len(shift_vals) - 1 or (i >= zero_idx - 3 and i <= zero_idx + 3):
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
    
    def sortEstPeaks(self, steepness, peaks, shift_vals):
        # sort peaks
        sorted_peaks = [x for _, x in sorted(zip(steepness, peaks), reverse=True)]
        top_peaks = sorted_peaks[:2]
        peak_shifts = [shift_vals[peak] for peak in top_peaks]
        if abs(peak_shifts[0]) == abs(peak_shifts[1]):
            estimated_shift = abs(peak_shifts[0])
        else:
            # raise error
            print("Error: Multiple peaks detected")
            estimated_shift = None
        return estimated_shift
    
    def computePixelShift(self):
        if self.grey:
            shift_vals, corr_vals = self.computeCrossCorrelation(self.It)
            corr_vals = self.applyFilter(corr_vals)
            peaks, steepness = self.obtainCorrelationPeaks(shift_vals, corr_vals)
            estimated_shift = self.sortEstPeaks(steepness, peaks, shift_vals)
        else:
            plt.figure()
            est_shifts = []
            for i in range(self.It.shape[2]):
                # convert this channel to greyscale
                # normalise self.It
                channel_i_grey = self.It[:, :, i] / np.max(self.It[:, :, i])
                shift_vals, corr_vals = self.computeCrossCorrelation(channel_i_grey)
                corr_vals = self.applyFilter(corr_vals)
                peaks, steepness = self.obtainCorrelationPeaks(shift_vals, corr_vals)
                # plot corr val
                
                plt.plot(shift_vals, corr_vals, label=f"Channel {i}")
                estimated_shift = self.sortEstPeaks(steepness, peaks, shift_vals)
                est_shifts.append(estimated_shift)
            plt.legend()
            # plt.show()
            estimated_shift = int(np.mean(est_shifts))
        
        self.estimated_shift = estimated_shift

    def getImagePSF(self, w1_est):

        psf = np.zeros(self.estimated_shift + 1)
        psf[-1] = w1_est
        psf[0] = 1 - w1_est

        psf = psf/np.sum(psf)

        psf = np.expand_dims(psf, axis=0)
        self.estimated_psf = psf
        # print(psf)

    def testPSF(self, img):
        new_blur = sp.ndimage.convolve(img, self.estimated_psf, mode='constant')

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

        # plt.show()
        return new_blur
    
    def loss_func(self, est, img):
        w1_est = est

        # img = self.It.copy()

        # Contrast enhance image
        if np.max(img) > 1:
            img = img / np.max(img)

        # img = sp.ndimage.shift(img, shift=(0, -self.estimated_shift//2), mode='constant', cval=0)

        # Compute PSF and perform Wiener deconvolution
        self.getImagePSF(w1_est)
        # print(self.estimated_psf)

        deconvolved = sk.restoration.wiener(img, self.estimated_psf, balance=0)

        # print(f"deconvolved: {deconvolved}")

        # Use your computeCrossCorrelation function
        shift_vals, corr_vals = self.computeCrossCorrelation(deconvolved)

        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # plt.imshow(deconvolved, cmap='gray')
        # plt.title('Deconvolved Image')

        # plt.subplot(1, 2, 2)
        # plt.plot(shift_vals, corr_vals)
        # plt.title('Cross Correlation')
        # plt.xlabel('Shift Values')
        # plt.ylabel('Correlation Values')

        # plt.show()

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

        #print mse
        # print(f"mse: {np.mean((img - blur_rec)**2)*100}")
        savgol_filt = self.applyFilter(corr_vals)
        posidx = np.where(shift_vals == self.estimated_shift)[0][0]
        negidx = np.where(shift_vals == -self.estimated_shift)[0][0]
        loss += abs(savgol_filt[posidx]) + abs(savgol_filt[negidx])        

        central_region_idx = np.where((shift_vals > 3) & (-shift_vals < -3))[0]
        central_region_vals = savgol_filt[central_region_idx]
        flatness_loss = np.std(central_region_vals)
        loss += flatness_loss**2
        # print(f"Initial loss: {loss}")
        # print("test")
        if self.min_loss is None or loss < self.min_loss:
            print(f"Newest loss is {loss} for w1: {w1_est}")
            self.min_loss = loss
            self.best_w1 = w1_est
        # print(f"final loss: {loss} with w1: {w1_est}")
        # plot corr vals and filtered corr
        # plt.figure()
        # plt.imshow(deconvolved, cmap='gray')
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # plt.plot(shift_vals, corr_vals)
        # plt.title('Cross Correlation')
        # plt.xlabel('Shift Values')
        # plt.ylabel('Correlation Values')
        # plt.subplot(1, 2, 2)
        # plt.plot(shift_vals, savgol_filt)
        # plt.title('Filtered Cross Correlation')
        # plt.xlabel('Shift Values')
        # plt.ylabel('Correlation Values')
        # plt.show()

        if loss <= 2e-4:
            raise EarlyStoppingException(f"Early stopping: loss has reached the threshold of 2e-6")


        return loss

    
    def opt_minimise_weights(self, w1guess, bounds, method, img):
        try:
            result = sp.optimize.differential_evolution(self.loss_func, bounds, args=(img,))
            w1_est_global = result.x
            print(f"Global minimum: {w1_est_global}")
            # result = sp.optimize.minimize(self.loss_func, w1guess, args=(img), bounds=bounds, method=method)#, options={'xtol': 1e-8, 'ftol': 1e-8, 'maxiter': 10000})#"L-BFGS-B")
            return result.x
        except EarlyStoppingException as e:
            print(e)
            return self.best_w1
    
    def deconvolve(self, shifted):
        # check if shifted is between 0 and 1 if not convert
        if np.max(shifted) > 1:
            shifted = shifted / np.max(shifted)
        
        deconvolved = sk.restoration.wiener(shifted, self.estimated_psf, balance=0)
        # normalise deconvolved
        deconvolved = deconvolved / np.max(deconvolved)
        return deconvolved
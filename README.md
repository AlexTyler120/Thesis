# Thesis
As of 22/08 this is where we are at..
Image.py takes in an original images resizes whatever all good.
Shift image
Shift image initialises with the double image whether it is grey or not
To compute the pixel shift first take the cross correlation. use valid for speed but only need the max peaks. Peaks represent x values where the image has been shifted over and it overlaps.
A savgol filter is then applied. The smoothed signal is subtracted from the original signal to highlight these peaks. Works well so far for all images.
Once the savgol filtered line is subtracted we can take the result and measure the peaks and their steepness. The steepest apart from the the center peak represent the value by which the image has been shifted.
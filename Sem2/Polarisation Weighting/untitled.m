close all;
clear;
% Read the image
img = imread('patch.png'); % Change to your image filename
img = rgb2gray(img); % Convert to grayscale if the image is RGB

% Get image dimensions
[imgHeight, imgWidth] = size(img);
img = img(5:imgHeight-5, 5:imgWidth-5);
[imgHeight, imgWidth] = size(img);
% Prepare for normalized cross-correlation
shifts = -floor(imgWidth/2) : floor(imgWidth/2);
xcorrValues = zeros(size(shifts));

% Loop through shifts and compute normalized cross-correlation
for shiftIdx = 1:length(shifts)
    shift = shifts(shiftIdx);
    
    % Shift the image horizontally (cyclic shift)
    % img_shifted = circshift(img, [0, shift]);
    % Apply constant shift by padding the image with zeros
    if shift > 0
        % Shift right: pad left with zeros
        shifted_img = [ones(size(img, 1), shift), img(:, 1:end-shift)];
    elseif shift < 0
        % Shift left: pad right with zeros
        shifted_img = [img(:, -shift+1:end), ones(size(img, 1), -shift)];
    else
        % No shift
        shifted_img = img;
    end
    figure;
    imshow(shifted_img)

    % Perform normalized cross-correlation using normxcorr2
    c = normxcorr2(shifted_img, img);
    
    % Extract the peak of the cross-correlation result (normalized value)
    [~, maxIdx] = max(c(:));
    xcorrValues(shiftIdx) = c(maxIdx);
end
figure;
imshow(img);

% Plot the cross-correlation result
figure;
plot(shifts, xcorrValues, 'LineWidth', 2);
xlabel('Shift (pixels)');
ylabel('Normalized Cross-Correlation');
title('Normalized Cross-Correlation of Image with Shifted Versions');
grid on;

Ioriginal = imread("matlab/flower.jpg");
% resize to 256x256
I = imresize(Ioriginal, [256, 256]);
% convert to grayscale
I = rgb2gray(I);

I1 = I;
I2 = I;

% weight I1 by 0.3
I1 = I1 * 0.3;
% weight I2 by 0.7
I2 = I2 * 0.7;

shift = 10;

I2(:, shift:end) = I2(:, 1:end-shift+1);

It = I1 + I2;

% Define the shift value
shift = 10;

% Create the PSF as a line of length `shift`
PSF = zeros(1, shift);
PSF(shift) = 0.7; % Since I2 was weighted by 0.7
PSF(1) = 0.3; % Since I1 was weighted by 0.3

% Normalize the PSF
PSF = PSF / sum(PSF);

% Convert the PSF to a 2D matrix (if necessary, depending on the blur's direction)
PSF = repmat(PSF, [1, 1]); % No change needed here since blur is 1D (horizontal)

% Apply the PSF to the original image using imfilter
I_blurred = imfilter(double(I), PSF, 'conv', 'same');

% Remove the first 10 rows and columns to match the size of It

% Compare I_blurred_cropped with It
It_double = double(It);
I_blurred_double = double(I_blurred);

% Now subtract and display the difference image as a heatmap in color
figure;
subplot(1, 3, 1);
imshow(It, []);
title('Blurred Image (It)');

subplot(1, 3, 2);
imshow(I_blurred_double, []);
title('Image Blurred with PSF');

subplot(1, 3, 3);
heatmap(abs(It_double - I_blurred_double));
title('Difference Image');

% Perform Wiener deconvolution
estimated_original_image = deconvwnr(It_double, PSF);

% Display the estimated original image
figure;
imshow(estimated_original_image, []);
title('Estimated Original Image');

% heatmap difference between estimated original image and original image
% Calculate the difference between the estimated original image and the original cropped image
difference_image = abs(estimated_original_image - double(I));

% Create a heatmap with a more colorful colormap and less grid visibility
figure;
h = heatmap(difference_image, 'Colormap', parula, 'GridVisible', 'off', 'ColorLimits', [0, max(difference_image(:))]);
title('Difference between Estimated Original Image and Original Image');
% Adjust the heatmap properties
h.XDisplayLabels = nan(size(h.XData)); % Remove x-axis labels
h.YDisplayLabels = nan(size(h.YData)); % Remove y-axis labels
% Optionally, you can adjust the color limits or colormap further for better visualization
h.ColorbarVisible = 'on'; % Show the colorbar for reference
w1 = 0.3; w2 = 0.7; shift = 5; name = 'nopol.jpg';

[It, image] = create_image(w1, w2, shift, name);


% fprintf('Estimated Shift: %d\n', est_shift);
[shift] = estimate_shift(It);



% psf = create_psf(w1, w2, est_shift);

% blurred = psf_test(psf, It, image);

% % Perform Wiener deconvolution
% estimated_original_image = deconvwnr(It, psf);

% % subplot
% figure;
% subplot(1, 3, 1);
% imshow(image, []);
% title('Original Image');
% subplot(1, 3, 2);
% imshow(estimated_original_image, []);
% title('Estimated Original Image');

% subplot(1, 3, 3);
% difference_image = abs(estimated_original_image - image);
% h = heatmap(difference_image, 'Colormap', parula, 'GridVisible', 'off', 'ColorLimits', [0, max(difference_image(:))]);
% title('Difference between Estimated Original Image and Original Image');
% % Adjust the heatmap properties
% h.XDisplayLabels = nan(size(h.XData)); % Remove x-axis labels
% h.YDisplayLabels = nan(size(h.YData)); % Remove y-axis labels
% % Optionally, you can adjust the color limits or colormap further for better visualization
% h.ColorbarVisible = 'on'; % Show the colorbar for reference


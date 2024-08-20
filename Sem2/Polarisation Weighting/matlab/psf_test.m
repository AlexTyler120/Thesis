function blurred = psf_test(psf, original_blurred, original)
    blurred = imfilter(original, psf, 'conv', 'same');

    figure;
    subplot(1, 3, 1);
    imshow(original_blurred, []);
    title('Original Blurred Image');

    subplot(1, 3, 2);
    imshow(blurred, []);
    title('New Blurred Image');

    difference = abs(double(original_blurred) - double(blurred));
    disp(max(difference(:)));
    subplot(1, 3, 3);
    h = heatmap(difference, 'Colormap', parula, 'GridVisible', 'off', 'ColorLimits', [0, 256]);
    title('Difference between Original blurred and estimated blurred');
    % Adjust the heatmap properties
    h.XDisplayLabels = nan(size(h.XData)); % Remove x-axis labels
    h.YDisplayLabels = nan(size(h.YData)); % Remove y-axis labels
    % Optionally, you can adjust the color limits or colormap further for better visualization
    h.ColorbarVisible = 'on'; % Show the colorbar for reference
end
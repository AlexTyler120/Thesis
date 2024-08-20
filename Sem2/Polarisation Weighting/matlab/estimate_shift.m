function [shift_values, correlation_values] = estimate_shift(image)
    [~, width, ~] = size(image);
    % max_shift = floor(width / 10);
    max_shift = 25;

    shift_values = -max_shift:max_shift;
    correlation_values = zeros(1, length(shift_values));

    % Loop over possible shifts
    for i = 1:length(shift_values)
        x_shift = shift_values(i);

        % Apply shift to the image channel along the x-axis
        image_shifted = circshift(image, [0 x_shift]);

        % Compute cross-correlation using xcorr
        [cross_corr, lags] = xcorr(image(:), image_shifted(:));
        
        % Find the correlation value at zero shift (when lag is 0)
        zero_shift_idx = find(lags == 0);
        correlation_values(i) = cross_corr(zero_shift_idx);
    end
    % plot
    % plot(shift_values, correlation_values)
    filtered_corr = baseline_filter(correlation_values);
    plot(shift_values, filtered_corr)
    steep = corr_peaks(shift_values, filtered_corr);
    [maxVal,maxI] = max(steep,[],"linear");
    % nan 0 val
    steep(maxI) = NaN;
    % get next two
    [maxVal,maxI] = max(steep,[],"linear");
    shift_values(maxI)
    steep(maxI) = NaN;
    [maxVal,maxI] = max(steep,[],"linear");
    shift_values(maxI)
end

%% savgol baseline reduct func
function output = baseline_filter(correlations)
    gauss = smoothdata(correlations, 'gaussian', 3);
    golayd = smoothdata(correlations, 'sgolay', 'Degree', 3 );
    % window_l = 5;
    % poly = 3;
    % baseline = sgolayfilt(correlations, poly, window_l);
    hol = correlations - golayd;
    output = correlations - gauss + hol;
end

%% Obtain peaks
function steepness = corr_peaks(shift, correlation)
    
    % peaks = zeros(1, length(shift));
    steepness = zeros(1, length(shift));

    for i=1:length(shift)
        if i == 1 || i == length(shift) || i == (length(shift) - 1)/2
            continue
        else
            left_corr = correlation(i-1);
            right_corr = correlation(i+1);
            mid_corr = correlation(i);
            left_rise = mid_corr - left_corr;
            right_rise = mid_corr - right_corr;
            if left_rise > 0 && right_rise > 0
                steepness(i) = left_rise + right_rise;
            end
        end
    end
end
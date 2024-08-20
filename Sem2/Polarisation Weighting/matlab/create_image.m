function [I, I_o] = create_image(w1, w2, shift, name)
    image = imread(name);
    % resize image to 10%
    image = imresize(image, 0.15);

    image = rgb2gray(image);
    % rotate 90 deg left
    image = imrotate(image, -90);

    I1 = image;
    I2 = image;

    I1 = I1 * w1;
    I2 = I2 * w2;

    I2(:, shift:end) = I2(:, 1:end-shift+1);

    I = I1 + I2;
    I_o = image;
end
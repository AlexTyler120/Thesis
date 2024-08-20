function psf = create_psf(w1, w2, shift)
    psf = zeros(1, shift);
    psf(shift) = w2;
    psf(1) = w1;

    % normalise
    psf = psf / sum(psf);

    % convert to 2D
    psf = repmat(psf, [1, 1]);
    
end
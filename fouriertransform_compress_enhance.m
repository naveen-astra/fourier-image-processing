%compression
% Read and display the original image
originalImage = imread('D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg');
imshow(originalImage);
title('Original Image');

% Perform Fourier transform
fourierImage = fft2(rgb2gray(originalImage));

% Perform inverse Fourier transform to reconstruct the image
reconstructedImage = ifft2(fourierImage);
reconstructedImage = uint8(abs(reconstructedImage));

% Display the reconstructed image
figure;
imshow(reconstructedImage);
title('Reconstructed Image');

% Calculate compression ratio
originalSize = numel(originalImage);
compressedSize = numel(fourierImage);
compressionRatio = originalSize / compressedSize;
disp(['Compression Ratio: ', num2str(compressionRatio)]);

%%
%enhancement
% Read the original image
originalImage = imread('D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg');

% Display the original image in a separate window
figure;
imshow(originalImage);
title('Original Image');

% Convert the image to grayscale
grayImage = rgb2gray(originalImage);

% Display the grayscale image in a separate window
figure;
imshow(grayImage);
title('Grayscale Image');

% Perform Fourier transform
fourierImage = fftshift(fft2(double(grayImage)));

% Define a high-pass filter (e.g., Laplacian filter)
[M, N] = size(grayImage);
u = 0:(M-1);
v = 0:(N-1);
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2);
cutoffFrequency = 0.1; % Adjust cutoff frequency as needed
H = double(D > cutoffFrequency);

% Apply the high-pass filter in the frequency domain
filteredFourierImage = fourierImage .* H;

% Perform inverse Fourier transform to obtain the enhanced image
filteredImage = ifft2(ifftshift(filteredFourierImage));
enhancedImage = uint8(abs(filteredImage));

% Display the enhanced image in a separate window
figure;
imshow(enhancedImage);
title('Enhanced Image');

% Convert the enhanced image to color using a colormap
coloredEnhancedImage = ind2rgb(enhancedImage, jet(256));

% Display the colored enhanced image in a separate window
figure;
imshow(coloredEnhancedImage);
title('Colored Enhanced Image');

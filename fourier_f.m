%compression
originalImage = imread('img');
imshow(originalImage);
title('Original Image');
fourierImage = fft2(rgb2gray(originalImage));
reconstructedImage = ifft2(fourierImage);
reconstructedImage = uint8(abs(reconstructedImage));
figure;
imshow(reconstructedImage);
title('Reconstructed Image');
originalSize = numel(originalImage);
compressedSize = numel(fourierImage);
compressionRatio = originalSize / compressedSize;
disp(['Compression Ratio: ', num2str(compressionRatio)]);

%%
%enhancement
originalImage = imread('img');
figure;
imshow(originalImage);
title('Original Image');
grayImage = rgb2gray(originalImage);
figure;
imshow(grayImage);
title('Grayscale Image');
fourierImage = fftshift(fft2(double(grayImage)));
[M, N] = size(grayImage);
u = 0:(M-1);
v = 0:(N-1);
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2);
cutoffFrequency = 0.1; 
H = double(D > cutoffFrequency);
filteredFourierImage = fourierImage .* H;
filteredImage = ifft2(ifftshift(filteredFourierImage));
enhancedImage = uint8(abs(filteredImage));
figure;
imshow(enhancedImage);
title('Enhanced Image');
coloredEnhancedImage = ind2rgb(enhancedImage, jet(256));
figure;
imshow(coloredEnhancedImage);
title('Colored Enhanced Image');

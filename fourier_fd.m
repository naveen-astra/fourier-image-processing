%compression
originalImage = imread('https://4kwallpapers.com/images/wallpapers/landscape-windows-11-lake-forest-day-time-2560x1080-8621.jpeg');
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
originalImage = imread('https://4kwallpapers.com/images/wallpapers/landscape-windows-11-lake-forest-day-time-2560x1080-8621.jpeg');
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
%%
%restoration
originalImage = imread('https://images.wallpaperscraft.com/image/single/mountain_lake_landscape_79402_2560x1600.jpg');
originalImage = im2double(originalImage);
figure;
imshow(originalImage);
title('Original Image');
LEN = 21;
THETA = 11;
motionBlurFilter = fspecial('motion', LEN, THETA);
blurredImage = imfilter(originalImage, motionBlurFilter, 'conv', 'symmetric');
noisyBlurredImage = imnoise(blurredImage, 'gaussian', 0, 0.01);
figure;
imshow(noisyBlurredImage);
title('Blurred and Noisy Image');
F_noisyBlurred = fft2(noisyBlurredImage);
H = psf2otf(motionBlurFilter, size(originalImage));
epsilon = 1e-3;
F_restored = F_noisyBlurred ./ (H + epsilon);
restoredImage = ifft2(F_restored);
restoredImage = real(restoredImage);
figure;
imshow(restoredImage, []);
title('Restored Image using Fourier Transform');
%%
% feature extraction
img1 = imread('https://images.unsplash.com/photo-1549558549-415fe4c37b60?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8Mnx8fGVufDB8fHx8fA%3D%3D');
img2 = imread('https://4kwallpapers.com/images/wallpapers/landscape-windows-11-lake-forest-day-time-2560x1080-8621.jpeg');

img1 = rgb2gray(img1);
img2 = rgb2gray(img2);

% Resize
img1 = imresize(img1, [256 256]);
img2 = imresize(img2, [256 256]);

%Fourier Transform
F1 = fft2(img1);
F2 = fft2(img2);

% Shift the zero-frequency component to the center
F1 = fftshift(F1);
F2 = fftshift(F2);

% Compute the magnitude spectrum
magnitude1 = abs(F1);
magnitude2 = abs(F2);

% Display the magnitude spectra
figure;
subplot(1, 2, 1);
imshow(log(1 + magnitude1), []);
title('Magnitude Spectrum of Image 1');

subplot(1, 2, 2);
imshow(log(1 + magnitude2), []);
title('Magnitude Spectrum of Image 2');
feature1 = magnitude1(:);
feature2 = magnitude2(:);
distance = norm(feature1 - feature2);
disp(['Euclidean Distance between images: ', num2str(distance)]);

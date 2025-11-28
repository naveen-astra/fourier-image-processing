% Read the image
originalImage = imread('D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg');

% Convert the image to grayscale if needed
grayImage = rgb2gray(originalImage);

% Perform Fourier transform
fourierImage = fft2(double(grayImage));

% Display the magnitude spectrum of the Fourier transform
magnitudeSpectrum = abs(fourierImage);
logMagnitudeSpectrum = log(1 + magnitudeSpectrum);
imshow(logMagnitudeSpectrum, []);

% Perform inverse Fourier transform to reconstruct the image
reconstructedImage = ifft2(fourierImage);
reconstructedImage = uint8(abs(reconstructedImage));

% Display the original image and the reconstructed image
subplot(1, 2, 1);
imshow(grayImage);
title('Original Image');

subplot(1, 2, 2);
imshow(reconstructedImage);
title('Reconstructed Image');

% Calculate compression ratio
originalSize = numel(grayImage);
compressedSize = numel(fourierImage);
compressionRatio = originalSize / compressedSize;
disp(['Compression Ratio: ', num2str(compressionRatio)]);

%%
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
%50compress
% Read and display the original image
originalImage = imread("D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg");
imshow(originalImage);
title('Original Image');

% Perform Fourier transform
fourierImage = fft2(rgb2gray(originalImage));

% Define the compression ratio you want (e.g., 0.5 for 50% compression)
compressionRatio = 0.5;

% Calculate the number of coefficients to keep based on compression ratio
totalCoefficients = numel(fourierImage);
coefficientsToKeep = round(compressionRatio * totalCoefficients);

% Sort the Fourier coefficients by magnitude and keep only the top coefficients
sortedCoefficients = sort(abs(fourierImage(:)), 'descend');
thresholdValue = sortedCoefficients(coefficientsToKeep);
truncatedFourierImage = fourierImage .* (abs(fourierImage) >= thresholdValue);

% Perform inverse Fourier transform to reconstruct the image
reconstructedImage = ifft2(truncatedFourierImage);
reconstructedImage = uint8(abs(reconstructedImage));

% Display the reconstructed image
figure;
imshow(reconstructedImage);
title('Reconstructed Image');

% Calculate the actual compression ratio achieved
actualCoefficients = nnz(abs(truncatedFourierImage));
actualCompressionRatio = actualCoefficients / totalCoefficients;
disp(['Actual Compression Ratio: ', num2str(actualCompressionRatio)]);

%%

%enhance
% Read and display the original image
originalImage = imread("D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg");
imshow(originalImage);
title('Original Image');

% Convert the image to grayscale if needed
grayImage = rgb2gray(originalImage);

% Perform Fourier transform
fourierImage = fft2(double(grayImage));

% Create and apply a high-pass filter in the frequency domain
highPassFilter = fspecial('laplacian', 0.5);
filteredImage = fourierImage .* highPassFilter;

% Perform inverse Fourier transform to reconstruct the enhanced image
enhancedImage = ifft2(filteredImage);
enhancedImage = uint8(abs(enhancedImage));

% Display the enhanced image
figure;
imshow(enhancedImage);
title('Enhanced Image');

%%
%enhance2
% Read the original image
originalImage = imread("D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg");
imshow(originalImage);
title('Original Image');

% Convert the image to grayscale if needed
grayImage = rgb2gray(originalImage);

% Perform Fourier transform
fourierImage = fft2(double(grayImage));

% Create a high-pass filter (e.g., Laplacian filter)
filterSize = 5; % Adjust filter size as needed
highPassFilter = fspecial('laplacian', filterSize);

% Apply the filter in the frequency domain
filteredImage = fourierImage .* highPassFilter;

% Perform inverse Fourier transform to reconstruct the enhanced image
enhancedImage = ifft2(filteredImage);
enhancedImage = uint8(abs(enhancedImage));

% Display the enhanced image
figure;
imshow(enhancedImage);
title('Enhanced Image');
%%
%enhance3
% Read and display the original image
originalImage = imread('D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg');
imshow(originalImage);
title('Original Image');

% Convert the image to grayscale if needed
grayImage = rgb2gray(originalImage);

% Perform Fourier transform
fourierImage = fft2(double(grayImage));

% Create a high-pass filter manually
[M, N] = size(grayImage);
cutoffFrequency = 0.1; % Adjust cutoff frequency as needed
u = 0:(M-1);
v = 0:(N-1);
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2);
H = double(D > cutoffFrequency);

% Apply the high-pass filter in the frequency domain
filteredImage = fourierImage .* H;

% Perform inverse Fourier transform to reconstruct the enhanced image
enhancedImage = ifft2(filteredImage);
enhancedImage = uint8(abs(enhancedImage));

% Display the enhanced image
figure;
imshow(enhancedImage);
title('Enhanced Image');

%%
%enhance4
% Read and display the original image
originalImage = imread('D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg');
imshow(originalImage);
title('Original Image');

% Convert the image to grayscale
grayImage = rgb2gray(originalImage);

% Apply Fourier transform
fourierImage = fft2(double(grayImage));

% Define the high-pass filter in the frequency domain
cutoffFrequency = 0.1;
[M, N] = size(grayImage);
u = 0:(M-1);
v = 0:(N-1);
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2);
H = double(D > cutoffFrequency);

% Apply the high-pass filter
filteredImage = fourierImage .* H;

% Reconstruct the enhanced image
enhancedImage = ifft2(filteredImage);
enhancedImage = uint8(abs(enhancedImage));

% Display the enhanced image
figure;
imshow(enhancedImage);
title('Enhanced Image');

%%
%enhance5
% Read and display the original image
originalImage = imread('D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg');
imshow(originalImage);
title('Original Image');

% Convert the image to grayscale
grayImage = rgb2gray(originalImage);

% Apply Fourier transform
fourierImage = fft2(double(grayImage));

% Define the high-pass filter in the frequency domain
cutoffFrequency = 0.1;
[M, N] = size(grayImage);
u = 0:(M-1);
v = 0:(N-1);
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2);
H = double(D > cutoffFrequency);

% Apply the high-pass filter
filteredImage = fourierImage .* H;

% Reconstruct the enhanced image
enhancedImage = ifft2(filteredImage);
enhancedImage = uint8(abs(enhancedImage));

% Convert the enhanced image to RGB format
coloredEnhancedImage = cat(3, enhancedImage, enhancedImage, enhancedImage);

% Display the colored enhanced image
figure;
imshow(coloredEnhancedImage);
title('Colored Enhanced Image');

%%
%enhance6
% Read and display the original image
originalImage = imread('D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg');
imshow(originalImage);
title('Original Image');

% Convert the image to grayscale
grayImage = rgb2gray(originalImage);

% Apply Fourier transform
fourierImage = fft2(double(grayImage));

% Define the high-pass filter in the frequency domain
cutoffFrequency = 0.1;
[M, N] = size(grayImage);
u = 0:(M-1);
v = 0:(N-1);
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2);
H = double(D > cutoffFrequency);

% Apply the high-pass filter
filteredImage = fourierImage .* H;

% Reconstruct the enhanced image
enhancedImage = ifft2(filteredImage);
enhancedImage = uint8(abs(enhancedImage));

% Apply colorization
coloredEnhancedImage = ind2rgb(enhancedImage, jet(256));

% Display the colored enhanced image
figure;
imshow(coloredEnhancedImage);
title('Colored Enhanced Image');
%%
%enhance7
% Read the original image
originalImage = imread('D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg');

% Convert the image to grayscale
grayImage = rgb2gray(originalImage);

% Enhance the grayscale image using histogram equalization
enhancedGrayImage = histeq(grayImage);

% Apply contrast enhancement to the enhanced grayscale image
contrastEnhancedImage = imadjust(enhancedGrayImage);

% Apply color mapping to the contrast-enhanced image
coloredEnhancedImage = ind2rgb(contrastEnhancedImage, jet(256));

% Display the original image and the colored enhanced image
figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 2, 2);
imshow(coloredEnhancedImage);
title('Colored Enhanced Image');

%%
%enhance8
% Read the original image
originalImage = imread('D:\Amrita Class\Sem2\MFC 2 - Neethu Mohan\Project\starship super heavy.jpg');

% Convert the image to grayscale
grayImage = rgb2gray(originalImage);

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

% Convert the enhanced image to color using a colormap
coloredEnhancedImage = ind2rgb(enhancedImage, jet(256));

% Display the original image, grayscale image, and colored enhanced image
figure;
subplot(1, 3, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 3, 2);
imshow(grayImage);
title('Grayscale Image');

subplot(1, 3, 3);
imshow(coloredEnhancedImage);
title('Colored Enhanced Image');

%%
%enhance9
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
%%
% Load two example images
img1 = imread('https://images.unsplash.com/photo-1549558549-415fe4c37b60?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8Mnx8fGVufDB8fHx8fA%3D%3D');
img2 = imread('https://4kwallpapers.com/images/wallpapers/landscape-windows-11-lake-forest-day-time-2560x1080-8621.jpeg');

% Convert to grayscale
img1 = rgb2gray(img1);
img2 = rgb2gray(img2);

% Resize images to a common size
img1 = imresize(img1, [256 256]);
img2 = imresize(img2, [256 256]);

% Compute the Fourier Transform
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

% Flatten the magnitude spectra to use as feature vectors
feature1 = magnitude1(:);
feature2 = magnitude2(:);

% Compare the features using a distance metric (e.g., Euclidean distance)
distance = norm(feature1 - feature2);
disp(['Euclidean Distance between images: ', num2str(distance)]);
%%
% URLs of the images
url1 = 'https://4kwallpapers.com/images/wallpapers/landscape-windows-11-lake-forest-day-time-2560x1080-8621.jpeg'; % Main image URL
url2 = 'https://static.vecteezy.com/system/resources/previews/033/374/947/large_2x/sunset-on-the-beach-mountains-water-landscape-nature-hd-wallpaper-ai-generated-free-photo.jpg'; % Template image URL

% Read the images from the web
mainImg = imread('https://images.pexels.com/photos/346529/pexels-photo-346529.jpeg?cs=srgb&dl=pexels-bri-schneiter-28802-346529.jpg&fm=jpg');
templateImg = imread('https://static.vecteezy.com/system/resources/previews/033/374/947/large_2x/sunset-on-the-beach-mountains-water-landscape-nature-hd-wallpaper-ai-generated-free-photo.jpg');

% Convert to grayscale
mainImgGray = rgb2gray(mainImg);
templateImgGray = rgb2gray(templateImg);

% Resize template image to a common size
% Note: Adjust this if you want to keep the original template size
templateImgGray = imresize(templateImgGray, [64 64]);

% Compute the Fourier Transform
F_main = fft2(mainImgGray);
F_template = fft2(templateImgGray, size(mainImgGray, 1), size(mainImgGray, 2));

% Compute the Cross-Power Spectrum
R = F_main .* conj(F_template);
R = R ./ abs(R);

% Compute the Inverse Fourier Transform
corrResult = ifft2(R);
corrResult = fftshift(corrResult);

% Find the peak in the correlation result
[ypeak, xpeak] = find(abs(corrResult) == max(abs(corrResult(:))));

% Offset to account for the size of the template
yoffSet = ypeak - size(templateImgGray, 1) / 2;
xoffSet = xpeak - size(templateImgGray, 2) / 2;

% Display the result
figure;
imshow(mainImgGray);
hold on;
rectangle('Position', [xoffSet, yoffSet, size(templateImgGray, 2), size(templateImgGray, 1)], ...
          'EdgeColor', 'r', 'LineWidth', 2);
title('Detected Template');
%%
% Read the original image
originalImage = imread('https://images.wallpaperscraft.com/image/single/mountain_lake_landscape_79402_2560x1600.jpg');
originalImage = im2double(originalImage);

% Display the original image
figure;
imshow(originalImage);
title('Original Image');

% Simulate a motion blur
LEN = 21;
THETA = 11;
motionBlurFilter = fspecial('motion', LEN, THETA);

blurredImage = imfilter(originalImage, motionBlurFilter, 'conv', 'symmetric');

% Add Gaussian noise to the blurred image
noisyBlurredImage = imnoise(blurredImage, 'gaussian', 0, 0.01);

% Display the blurred and noisy image
figure;
imshow(noisyBlurredImage);
title('Blurred and Noisy Image');

% Perform Fourier Transform of the noisy blurred image
F_noisyBlurred = fft2(noisyBlurredImage);

% Perform Fourier Transform of the motion blur filter
H = psf2otf(motionBlurFilter, size(originalImage));

% Avoid division by zero by adding a small constant to H
epsilon = 1e-3;

% Perform inverse filtering in the frequency domain
F_restored = F_noisyBlurred ./ (H + epsilon);

% Perform Inverse Fourier Transform to get the restored image
restoredImage = ifft2(F_restored);
restoredImage = real(restoredImage);

% Display the restored image
figure;
imshow(restoredImage, []);
title('Restored Image using Fourier Transform');








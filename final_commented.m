% Read the original image from the URL and display it
originalImage = imread('https://4kwallpapers.com/images/wallpapers/landscape-windows-11-lake-forest-day-time-2560x1080-8621.jpeg');
imshow(originalImage);
title('Original Image');

% Convert the image to grayscale and apply the 2D Fourier transform
fourierImage = fft2(rgb2gray(originalImage));

% Apply the inverse 2D Fourier transform to reconstruct the image
reconstructedImage = ifft2(fourierImage); 

% Convert the reconstructed image to uint8 format and display it
reconstructedImage = uint8(abs(reconstructedImage));
figure;
imshow(reconstructedImage);
title('Reconstructed Image');

% Calculate the compression ratio
originalSize = numel(originalImage); % Number of elements in the original image
compressedSize = numel(fourierImage); % Number of elements in the Fourier-transformed image
compressionRatio = originalSize / compressedSize; % Compression ratio
disp(['Compression Ratio: ', num2str(compressionRatio)]);
%%
% Read the original image from the URL and display it
originalImage = imread('https://4kwallpapers.com/images/wallpapers/landscape-windows-11-lake-forest-day-time-2560x1080-8621.jpeg');
figure;
imshow(originalImage);
title('Original Image');

% Convert the image to grayscale and display it
grayImage = rgb2gray(originalImage);
figure;
imshow(grayImage);
title('Grayscale Image');

% Apply the 2D Fourier transform to the grayscale image and shift the zero-frequency component to the center
fourierImage = fftshift(fft2(double(grayImage)));

% Get the size of the grayscale image
[M, N] = size(grayImage);

% Create frequency arrays for the u and v directions
u = 0:(M-1);
v = 0:(N-1);

% Adjust the frequency arrays to have negative frequencies in the second half
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;

% Create a meshgrid of frequency coordinates
[V, U] = meshgrid(v, u);

% Calculate the distance matrix from the frequency origin
D = sqrt(U.^2 + V.^2);

% Define a cutoff frequency for the high-pass filter (0.1 is a normalized value)
cutoffFrequency = 0.1;

% Create the high-pass filter: values are 1 if the distance is greater than the cutoff, otherwise 0
H = double(D > cutoffFrequency);

% Apply the high-pass filter to the Fourier-transformed image
filteredFourierImage = fourierImage .* H;

% Apply the inverse 2D Fourier transform to get the filtered image and shift the zero-frequency component back
filteredImage = ifft2(ifftshift(filteredFourierImage));

% Convert the filtered image to uint8 format (absolute value to remove any imaginary part) and display it
enhancedImage = uint8(abs(filteredImage));
figure;
imshow(enhancedImage);
title('Enhanced Image');

% Convert the enhanced image to RGB using a colormap (jet) and display it
coloredEnhancedImage = ind2rgb(enhancedImage, jet(256));
figure;
imshow(coloredEnhancedImage);
title('Colored Enhanced Image');

%%
%restoration
% Read the original image from the URL and convert it to double precision format for processing
originalImage = imread('https://images.wallpaperscraft.com/image/single/mountain_lake_landscape_79402_2560x1600.jpg');
originalImage = im2double(originalImage);

% Display the original image
figure;
imshow(originalImage);
title('Original Image');

% Define the motion blur parameters
LEN = 21; % Length of the motion blur in pixels
THETA = 11; % Angle of the motion blur in degrees

% Create the motion blur filter using the specified parameters
motionBlurFilter = fspecial('motion', LEN, THETA);

% Apply the motion blur filter to the original image using convolution
blurredImage = imfilter(originalImage, motionBlurFilter, 'conv', 'symmetric');

% Add Gaussian noise to the blurred image with mean 0 and variance 0.01
noisyBlurredImage = imnoise(blurredImage, 'gaussian', 0, 0.01);

% Display the blurred and noisy image
figure;
imshow(noisyBlurredImage);
title('Blurred and Noisy Image');

% Apply the 2D Fourier transform to the noisy blurred image
F_noisyBlurred = fft2(noisyBlurredImage);

% Convert the motion blur filter to the frequency domain
H = psf2otf(motionBlurFilter, size(originalImage));

% Define a small constant to avoid division by zero during restoration
epsilon = 1e-3;

% Restore the image using the inverse filtering method
% Divide the Fourier transform of the noisy blurred image by the frequency domain representation of the blur filter
F_restored = F_noisyBlurred ./ (H + epsilon);

% Apply the inverse 2D Fourier transform to get the restored image
restoredImage = ifft2(F_restored);

% Take the real part of the restored image and display it
restoredImage = real(restoredImage);
figure;
imshow(restoredImage, []);
title('Restored Image using Fourier Transform');
%%
% Feature extraction
% Read the first image from the URL
img1 = imread('https://images.unsplash.com/photo-1549558549-415fe4c37b60?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8Mnx8fGVufDB8fHx8fA%3D%3D');

% Read the second image from the URL
img2 = imread('https://4kwallpapers.com/images/wallpapers/landscape-windows-11-lake-forest-day-time-2560x1080-8621.jpeg');

% Convert the first image to grayscale
img1 = rgb2gray(img1);

% Convert the second image to grayscale
img2 = rgb2gray(img2);

% Resize the first image to 256x256 pixels
img1 = imresize(img1, [256 256]);

% Resize the second image to 256x256 pixels
img2 = imresize(img2, [256 256]);

% Apply the 2D Fourier transform to the first image
F1 = fft2(img1);

% Apply the 2D Fourier transform to the second image
F2 = fft2(img2);

% Shift the zero-frequency component to the center for the first image
F1 = fftshift(F1);

% Shift the zero-frequency component to the center for the second image
F2 = fftshift(F2);

% Compute the magnitude spectrum of the Fourier-transformed first image
magnitude1 = abs(F1);

% Compute the magnitude spectrum of the Fourier-transformed second image
magnitude2 = abs(F2);

% Display the magnitude spectrum of the first image
figure;
subplot(1, 2, 1);
imshow(log(1 + magnitude1), []);
title('Magnitude Spectrum of Image 1');

% Display the magnitude spectrum of the second image
subplot(1, 2, 2);
imshow(log(1 + magnitude2), []);
title('Magnitude Spectrum of Image 2');

% Flatten the magnitude spectrum of the first image to a vector
feature1 = magnitude1(:);

% Flatten the magnitude spectrum of the second image to a vector
feature2 = magnitude2(:);

% Calculate the Euclidean distance between the feature vectors of the two images
distance = norm(feature1 - feature2);

% Display the Euclidean distance between the images
disp(['Euclidean Distance between images: ', num2str(distance)]);

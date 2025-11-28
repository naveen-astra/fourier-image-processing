# Fourier Image Processing

A Fourier Transform–based image processing project demonstrating frequency-domain filtering, compression, restoration, and feature extraction. Includes FFT visualizations, motion blur simulation, Wiener filtering, and contrast analysis.

## Features

- **Fourier Transform Analysis**: Visualize and understand frequency-domain representations of images
- **Image Compression**: Demonstrate lossy compression using FFT coefficient thresholding
- **Image Enhancement**: Apply frequency-domain filtering for contrast and sharpness improvements
- **Motion Blur Simulation**: Create and analyze motion blur effects in the frequency domain
- **Wiener Filtering**: Implement restoration techniques for degraded images
- **Feature Extraction**: Extract meaningful features from frequency-domain data

## Project Structure

```
├── final_commented.m          # Main script with detailed comments
├── final_f.m                  # Final implementation
├── fourier_f.m                # Core Fourier transform functions
├── fourier_fd.m               # Frequency domain operations
├── fouriertransform_compress_enhance.m  # Compression and enhancement pipeline
├── Fourier_bases.mlx          # MATLAB Live Script for Fourier bases visualization
├── plotcircle.m               # Utility for circular frequency visualization
├── *.fig                      # MATLAB figure files with results
└── README.md
```

## MATLAB Figure Files

- `original.fig` - Original input image
- `compressed.fig`, `compressed50.fig` - Compression results at different levels
- `enhanced.fig`, `enhanced2.fig`, `enhanced3.fig` - Various enhancement outputs
- `en_original.fig`, `en_greyscale.fig` - Greyscale conversions
- `en_colored_enhanced.fig`, `en_enhanced.fig` - Color and enhanced versions

## Requirements

- MATLAB R2019b or later
- Image Processing Toolbox (recommended)

## Usage

1. Open MATLAB and navigate to the project directory
2. Run `final_commented.m` for a step-by-step walkthrough
3. Use `fouriertransform_compress_enhance.m` for the complete pipeline
4. Explore `Fourier_bases.mlx` for interactive Fourier basis visualizations

## Theory

The project demonstrates key concepts in frequency-domain image processing:

1. **2D Discrete Fourier Transform (DFT)**: Converting spatial domain images to frequency domain
2. **Frequency Filtering**: Low-pass, high-pass, and band-pass filtering
3. **Compression**: Removing high-frequency components for data reduction
4. **Enhancement**: Amplifying specific frequency bands for better contrast

## License

This project is for educational purposes.

## Author

Naveen Babu 
Kishore B
Koushal Reddy
Sai Charan

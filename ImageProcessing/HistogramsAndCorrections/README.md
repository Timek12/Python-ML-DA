# Image Processing with Histograms and Intensity Corrections

This project explores basic image processing techniques focused on manipulating and analyzing image histograms, including rescaling intensity, gamma correction, and generating color channel histograms for RGB images. The project uses various Python libraries, such as `matplotlib`, `numpy`, and `scikit-image`, to load, process, and visualize images.

## Project Overview

The project includes three main parts:

1. **Histogram Visualization** - Create line and bar plots for image histograms, including cumulative histograms.
2. **Intensity Rescaling and Gamma Correction** - Adjust the image intensity range and apply gamma correction for brightness control.
3. **Linear Intensity Transformation** - Apply a linear function to control image brightness using a look-up table (LUT).

## Directory Structure

```plaintext
image_processing_histograms_and_corrections/
├── images/            # Folder for images used in the project (e.g., 'cameraman.bmp', 'dark_image.png')
├── requirements.txt   # List of required Python packages
├── README.md          # Project overview and usage information
└── main.py            # Main script with all code

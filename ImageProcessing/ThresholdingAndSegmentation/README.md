# Image Processing with Thresholding and Segmentation

This project focuses on various image processing techniques for thresholding, segmentation, and background subtraction. Using the `scikit-image` library, the project applies different thresholding methods, morphological transformations, and color-based segmentation to analyze and process grayscale and color images.

## Project Overview

The project is divided into multiple tasks:

1. **Thresholding and Histogram Visualization** - Explore different thresholding methods (e.g., mean, minimum) and visualize histograms to distinguish foreground from background.
2. **Background Subtraction for OCR** - Use morphological operations to separate background and foreground in printed text images, making them suitable for Optical Character Recognition (OCR).
3. **Threshold-based Color Segmentation** - Apply different thresholds to identify specific regions in a blood smear image.
4. **Sky Segmentation** - Segment the sky in an image based on color averages to isolate background and foreground.

## Directory Structure

```plaintext
image_processing_thresholding_and_segmentation/
├── images/            # Folder for input images (e.g., 'gears1.png', 'printed_text.png', 'blood_smear.jpg', 'airbus.png')
├── requirements.txt   # List of required Python packages
├── README.md          # Project overview and usage instructions
└── main.py            # Main script with all code

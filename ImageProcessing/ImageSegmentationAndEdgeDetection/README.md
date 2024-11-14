# Image Segmentation and Edge Detection

This project performs image segmentation using thresholding techniques and edge detection methods. It includes specific tasks for identifying regions in medical images, detecting objects like bolts, and identifying airplanes by segmentation and edge detection.

## Project Structure

### Tasks

- **Brain and Tumor Segmentation**: Thresholding-based segmentation of a brain scan to identify brain and tumor regions and calculate the tumor’s percentage in the brain area.
- **Bolt Edge Detection**: Reduces image resolution, applies Gaussian blurring, and uses the Canny edge detector to highlight bolt edges.
- **Airplane Detection**: Uses thresholding and region properties to identify airplanes in an aerial image and labels them with bounding boxes.

### Requirements

To run this project, install the dependencies:
```bash
pip install -r requirements.txt

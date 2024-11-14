# Image Noise Processing

This project demonstrates various image processing techniques, focusing on adding different types of noise to images and evaluating denoising methods. The following noise types are explored: salt-and-pepper, Gaussian, uniform, and impulse noise. Denoising methods include median, mean, Gaussian filters, and vector median filtering.

## Project Structure

- **Task 1**: Apply salt-and-pepper, Gaussian, and uniform noise to grayscale images. Calculate the Normalized Mean Squared Error (NMSE) between the original and noisy images to evaluate noise impact.
- **Task 2**: Apply denoising techniques (median, mean, Gaussian) to the noisy images, then calculate and display NMSE values to evaluate filter performance.
- **Task 3**: Apply impulse noise to an RGB image, varying the percentage of affected pixels.
- **Task 4**: Calculate NMSE between the original and noisy RGB images.
- **Task 5**: Compare the effectiveness of standard median and vector median filtering in denoising impulse noise from RGB images.

## Requirements

Install required packages with:
```bash
pip install -r requirements.txt

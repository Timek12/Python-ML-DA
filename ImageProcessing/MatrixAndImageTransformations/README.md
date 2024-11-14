# Matrix and Image Transformations

This project provides several matrix and image transformation tasks, including flipping and rotating matrices, embedding images within frames, cropping, dividing, shuffling, and applying specific geometric transformations to images.

## Project Structure
- **`matrix_image_transformations.py`**: Main script containing matrix transformations, image manipulations, and visualizations.
- **`requirements.txt`**: Dependencies required for running the code.

## Features
1. **Matrix Transformations**:
   - Flip (horizontal and vertical)
   - Rotate by 90° (left and right) and 180°
   - Extend non-square matrices to squares and extract central square portions

2. **Image Transformations**:
   - Center an image in a blank frame and apply transformations: flipping and rotating
   - Crop an image to a square, divide it into a grid, shuffle the grid cells, and display the result
   - Apply a custom transformation that swaps pixels along diagonals for visual effects

## Getting Started
1. **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Script**:
    ```bash
    python matrix_image_transformations.py
    ```

3. **Expected Output**: The script displays a series of transformations applied to matrices and images, including flips, rotations, cropped squares, shuffled cells, and geometric effects.

## Example Usage
In the main script, you can generate matrices, manipulate images, and visualize the results:

```python
# Matrix transformation examples
matrix = np.arange(1, 45).reshape(9, 5)
print("Original matrix:\n", matrix)
print("Horizontal flip:\n", np.fliplr(matrix))
print("Rotate 90° Right:\n", np.rot90(matrix, k=-1))

# Image transformation examples
output_image = center_image('images/lena.png')
flipped_image = cv2.flip(output_image, 1)  # Horizontal flip
cropped_img = crop_image_to_square('images/obraz_1.jpeg')
shuffled_img = shuffle_image(divide_image(cropped_img, 3))

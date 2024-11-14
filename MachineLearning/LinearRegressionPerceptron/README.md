# Linear Regression Perceptron

This project demonstrates a simple **linear regression** model using a perceptron that learns a linear function. Given a set of sample data points generated with noise, the perceptron iteratively adjusts its weight and bias to minimize the error and approximate the underlying linear function.

## Project Structure
- **`linear_regression_perceptron.py`**: Main script containing data generation, perceptron model, training function, and visualization.
- **`requirements.txt`**: Dependencies required for running the code.

## Features
1. **Data Generation**: The script generates synthetic data points around a randomly generated linear function.
2. **Perceptron Training**: A single-layer perceptron model adjusts its parameters through gradient descent to fit the data.
3. **Visualization**: The script visualizes both the target line and the perceptron’s approximation at each training epoch.

## Getting Started
1. **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Script**:
    ```bash
    python linear_regression_perceptron.py
    ```

3. **Expected Output**: The plot shows the perceptron's progress in fitting the linear data with each epoch, aiming to match the original line.

## Example Usage
The main script generates 100 sample data points, visualizes them, and then trains the perceptron to approximate the linear function:

```python
data = generate_sample_data(100)
visualize_data(data)

perceptron = Perceptron()
perceptron.train(data)

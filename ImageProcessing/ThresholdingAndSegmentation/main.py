# zadanie 1
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold, threshold_mean, threshold_otsu, threshold_triangle
from skimage import data, io, color, filters, morphology
from skimage.morphology import disk, ball
import numpy as np
from skimage.filters.rank import maximum
from skimage.util import img_as_ubyte

img = io.imread('gears1.png', as_gray=True)

fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()

# mean, minimum

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
ax = axes.ravel()

ax[0] = plt.subplot(2, 2, 1)
ax[1] = plt.subplot(2, 2, 2)
ax[2] = plt.subplot(2, 2, 3)
ax[3] = plt.subplot(2, 2, 4)

thresh_mean = threshold_mean(img)
binary_mean = img > thresh_mean

ax[0].imshow(~binary_mean, cmap=plt.cm.gray)
ax[0].set_title('Mean method')

ax[1].hist(img.ravel(), bins=256)
ax[1].set_title('Histogram for Mean method')
ax[1].axvline(thresh_mean, color='r')
ax[1].set_xlabel('pixels')
ax[1].set_ylabel('brightness')

thresh_minimum = threshold_mean(img)
binary_minimum = img > thresh_minimum

ax[2].imshow(~binary_minimum, cmap=plt.cm.gray)
ax[2].set_title('Minimum method')

ax[3].hist(img.ravel(), bins=256)
ax[3].set_title('Histogram for Minimum method')
ax[3].axvline(thresh_minimum, color='r')
ax[3].set_xlabel('pixels')
ax[3].set_ylabel('brightness')

plt.show()

# zadanie 2

# Step 1: Load and invert the image
img = io.imread('printed_text.png', as_gray=True)
img_inverted = 1 - img

# Step 2: Test initial thresholding on inverted image
fig, ax = try_all_threshold(img_inverted, figsize=(10, 8), verbose=False)
plt.show()

# Step 3: Convert image to 8-bit for rank processing
img_8bit = img_as_ubyte(img)  # Convert to 8-bit

# Step 4: Apply maximum filter for background extraction with a larger disk size
disk_size = 50  # Adjust this as necessary for a smoother background
background = filters.rank.maximum(img_8bit, morphology.disk(disk_size))

# Step 5: Subtract the inverted image from the background
subtracted_image = img_as_ubyte(background) - img_as_ubyte(img_inverted)

# Display intermediate results
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.imshow(img_inverted, cmap='gray')
plt.title('Inverted Image')

plt.subplot(3, 1, 2)
plt.imshow(background, cmap='gray')
plt.title('Background (Maximum Filter)')

plt.subplot(3, 1, 3)
plt.imshow(subtracted_image, cmap='gray')
plt.title('Subtracted Image')
plt.show()

# Step 6: Thresholding on subtracted image
fig, ax = try_all_threshold(subtracted_image, figsize=(10, 8), verbose=False)
plt.show()

# Final threshold using Otsu's method
thresh_val = threshold_otsu(subtracted_image)
binary_result = subtracted_image > thresh_val

# Display final result
plt.imshow(binary_result, cmap='gray')
plt.title("Final Thresholded Image for OCR")
plt.show()


# zadanie 3
img = io.imread('blood_smear.jpg', as_gray=True)

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
ax = axes.ravel()

t1 = 0.9
img1 = img > t1

t2 = 0.5
img2 = img > t2

colored_img = np.ones((*img.shape, 3))

colored_img[img < t2] = [0, 0, 1]  # Blue
colored_img[(img > t2) & (img < t1)] = [1, 0, 0]  # Red

ax[0].imshow(img, plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(img1, plt.cm.gray)
ax[1].set_title('Image after threshold 0.9')

ax[2].imshow(img2, plt.cm.gray)
ax[2].set_title('Image after threshold 0.5')

ax[3].imshow(colored_img, plt.cm.gray)
ax[3].set_title('Colored Image based on Thresholds')

plt.show()

# samolot
airbus = io.imread('airbus.png', as_gray=False)

# Read the image as RGB
niebo = airbus[:75, :350]

# Calculate the average values for R, G, B channels
avg_r = np.average(niebo[:, :, 0])
avg_g = np.average(niebo[:, :, 1])
avg_b = np.average(niebo[:, :, 2])

# Create the average color vector
avg_color = np.array([avg_r, avg_g, avg_b])

# Initialize maxDist to zero
maxDist = 0

# Iterate over all pixels in the niebo image
for i in range(niebo.shape[0]):
    for j in range(niebo.shape[1]):
        # Get the current pixel color vector (ignore alpha channel if present)
        pixel_color = niebo[i, j, :3]

        # Calculate the distance between the average color and the current pixel color
        dist = np.linalg.norm(avg_color - pixel_color)

        # Update maxDist if the current distance is greater
        if dist > maxDist:
            maxDist = dist

# Read the airbus image as RGB

# Create a binary image to store the result
binary_airbus = np.zeros((airbus.shape[0], airbus.shape[1]), dtype=np.int8)

# Iterate over all pixels in the airbus image
for i in range(airbus.shape[0]):
    for j in range(airbus.shape[1]):
        # Get the current pixel color vector (ignore alpha channel if present)
        pixel_color = airbus[i, j, :3]

        # Calculate the distance between the average color and the current pixel color
        dist = np.linalg.norm(avg_color - pixel_color)

        if dist > maxDist:
            binary_airbus[i, j] = 1
        else:
            binary_airbus[i, j] = 0

# Display the binary image
plt.imshow(binary_airbus, cmap='gray')
plt.title('Image created by setting pixel values to the average value of the background.')
plt.show()
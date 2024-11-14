import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

M, N = 9, 5
matrix = np.arange(1, M * N + 1).reshape(M, N)
print('Original matrix: \n', matrix)

horizontal_flip = np.zeros_like(matrix)
for i in range(M):
    for j in range(N):
        horizontal_flip[i, j] = matrix[i, N - 1 - j]
print("Horizontal Flip (ver.1):\n", horizontal_flip)

horizontal_flip = np.fliplr(matrix)
print("Horizontal Flip: (ver.2)\n", horizontal_flip)


vertical_flip = np.zeros_like(matrix)
for i in range(M):
    for j in range(N):
        vertical_flip[i, j] = matrix[M - 1 - i, j]
print("Vertical Flip (ver.1):\n", vertical_flip)

vertical_flip = np.flipud(matrix)
print("Vertical Flip: (ver.2)\n", vertical_flip)

rotate_90_right = np.zeros((N, M), dtype=int)
for i in range(M):
    for j in range(N):
        rotate_90_right[j, M - 1 - i] = matrix[i, j]
print("Rotate 90° Right (ver.1):\n", rotate_90_right)

rotate_90_right = np.rot90(matrix, k=-1)
print("Rotate 90° Right(ver.2):\n", rotate_90_right)

rotate_90_left = np.zeros((N, M), dtype=int)
for i in range(M):
    for j in range(N):
        rotate_90_left[N - 1 - j, i] = matrix[i, j]
print("Rotate 90° Left (ver.1):\n", rotate_90_left)

rotate_90_left = np.rot90(matrix, k=1)
print("Rotate 90° Left (ver.2):\n", rotate_90_left)

rotate_180 = np.zeros_like(matrix)
for i in range(M):
    for j in range(N):
        rotate_180[i, j] = matrix[M - 1 - i, N - 1 - j]
print("Rotate 180° (ver.1):\n", rotate_180)

rotate_180 = np.rot90(matrix, k=2)
print("Rotate 180° (ver.2):\n", rotate_180)

if M > N:
    extended_matrix = np.zeros((M, M), dtype=int)
    extended_matrix[:, (M-N)//2:(M+N)//2] = matrix
    print("Extend the matrix to a square MxM:\n", extended_matrix)

if M > N:
    start_row = (M - N) // 2 # starting row
    end_row = start_row + N
    cut_matrix = matrix[start_row:end_row, :]
    print("Cut out a square NxN from the matrix:\n", cut_matrix)


image = cv2.imread('images/lena.png', cv2.IMREAD_GRAYSCALE)

output_image = np.zeros((480, 640), dtype=np.uint8)

x_offset = (640 - image.shape[1]) // 2
y_offset = (480 - image.shape[0]) // 2

output_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image

horizontal_flip = cv2.flip(output_image, 1)
vertical_flip = cv2.flip(output_image, 0)
rotate_90_right = cv2.rotate(output_image, cv2.ROTATE_90_CLOCKWISE)
rotate_90_left = cv2.rotate(output_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
rotate_180 = cv2.rotate(output_image, cv2.ROTATE_180)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(output_image, cmap='gray')
axes[0, 0].set_title('Original Image')

axes[0, 1].imshow(horizontal_flip, cmap='gray')
axes[0, 1].set_title('Horizontal Flip')

axes[0, 2].imshow(vertical_flip, cmap='gray')
axes[0, 2].set_title('Vertical Flip')

axes[1, 0].imshow(rotate_90_right, cmap='gray')
axes[1, 0].set_title('Rotate 90° Right')

axes[1, 1].imshow(rotate_90_left, cmap='gray')
axes[1, 1].set_title('Rotate 90° Left')

axes[1, 2].imshow(rotate_180, cmap='gray')
axes[1, 2].set_title('Rotate 180°')

plt.tight_layout()
plt.show()


image_path = "images/obraz_1.jpeg"
img = cv2.imread(image_path)

height, width, _ = img.shape

side_length = min(height, width)

start_x = (width - side_length) // 2
start_y = (height - side_length) // 2

cropped_img = img[start_y:start_y + side_length, start_x:start_x + side_length]
img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title('Cropped image')
plt.axis('off')
plt.show()

num_splits = 3

new_size = (side_length - side_length % num_splits, side_length - side_length % num_splits)
resized_img = cv2.resize(img_rgb, new_size)

def divide_image(image, num_splits):
    height, width, _ = image.shape
    grid_size_h = height // num_splits
    grid_size_w = width // num_splits

    sub_images = []

    for row in range(0, height, grid_size_h):
        for col in range(0, width, grid_size_w):
            sub_image = image[row:row + grid_size_h, col:col + grid_size_w]
            sub_images.append(sub_image)

    return sub_images

def shuffle_image(sub_images, num_splits):
    random.shuffle(sub_images)
    shuffled_img = np.zeros_like(resized_img)

    grid_size_h = resized_img.shape[0] // num_splits
    grid_size_w = resized_img.shape[1] // num_splits

    idx = 0
    for row in range(0, resized_img.shape[0], grid_size_h):
        for col in range(0, resized_img.shape[1], grid_size_w):
            shuffled_img[row:row + grid_size_h, col:col + grid_size_w] = sub_images[idx]
            idx += 1

    return shuffled_img

sub_images = divide_image(resized_img, num_splits)

shuffled_img = shuffle_image(sub_images, num_splits)

plt.figure(figsize=(6, 6))

original_img = cv2.imread('images/obraz_1.jpeg')
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

plt.imshow(original_img)
plt.title('Original image')
plt.axis('off')
plt.show()

plt.imshow(shuffled_img)
plt.title('Shuffled squares')
plt.axis('off')
plt.show()



def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape

    swap_sides = random.choice([True, False])
    swap_top_bottom = random.choice([True, False])

    new_img_array = np.copy(img)

    if swap_sides:
        for y in range(height):
            for x in range(width):
                left_diagonal = (y * width) / height
                right_diagonal = width - 1 - (y * width) / height

                if x <= left_diagonal and x <= right_diagonal:
                    new_x = width - 1 - x
                    new_y = height - 1 - y
                    new_img_array[y, x] = img[new_y, new_x]
                    new_img_array[new_y, new_x] = img[y, x]

    if swap_top_bottom:
        for y in range(height):
            for x in range(width):
                left_diagonal = (y * width) / height
                right_diagonal = width - 1 - (y * width) / height

                if x <= left_diagonal and x >= right_diagonal:
                    new_y = height - 1 - y
                    new_img_array[y, x] = img[new_y, x]
                    new_img_array[new_y, x] = img[y, x]

    return new_img_array

result_image = process_image('images/obraz_1.jpeg')

original_img = cv2.imread('images/obraz_1.jpeg')
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

plt.imshow(original_img)
plt.title('Original image')
plt.axis('off')
plt.show()

plt.imshow(result_image)
plt.title('Shuffled triangles')
plt.axis('off')
plt.show()


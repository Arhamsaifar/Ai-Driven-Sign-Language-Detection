import cv2
import os
import numpy as np

# List of all folders to augment
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N"]

def augment_image(img):
    augmented_images = []

    # Original
    augmented_images.append(img)

    # Flip horizontally
    augmented_images.append(cv2.flip(img, 1))

    # Rotate
    center = (img.shape[1] // 2, img.shape[0] // 2)
    for angle in [-15, 15]:
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))
        augmented_images.append(rotated)

    # Add Gaussian Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    augmented_images.append(blurred)

    # Brightness adjustment
    bright = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
    augmented_images.append(bright)

    return augmented_images

# Loop through each letter folder
for letter in letters:
    input_folder = f"Data/{letter}"
    output_folder = f"Data/{letter}_augmented"
    os.makedirs(output_folder, exist_ok=True)

    counter = 0
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(input_folder, filename)
            img = cv2.imread(path)

            augmented = augment_image(img)
            for i, aug in enumerate(augmented):
                cv2.imwrite(os.path.join(output_folder, f"aug_{counter}_{i}.jpg"), aug)

            counter += 1

print("✅ All Data Augmentation Completed for A–N!")

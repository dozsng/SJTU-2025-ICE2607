import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

images = ['img1.jpg', 'img2.jpg', 'img3.jpg']  # List the images
folder = "outputs"
os.makedirs(folder, exist_ok=True)  # Create outputs folder
# Using Numpy slicing
def color_split(img):
    b = img[:, :, 0]  # Blue
    g = img[:, :, 1]  # Green
    r = img[:, :, 2]  # Red
    return b, g, r


for img in images:
    img_path = 'images/' + img   # Get the path
    # Color Histogram
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    b, g, r = color_split(img_color)
    # Calculate color energy using Numpy
    blue_energy = np.sum(b)
    green_energy = np.sum(g)
    red_energy = np.sum(r)
    color_list = [blue_energy, green_energy, red_energy]
    total = sum(color_list)
    color_ratio = []
    for energy in color_list:
        ratio = float(energy / total)  # Calculate the ratio
        color_ratio.append(ratio)
    plt.bar(['Blue', 'Green', 'Red'], color_ratio, color=['blue', 'green', 'red'])
    for i, ratio in enumerate(color_ratio):
        plt.text(i, ratio + 0.003, f'{ratio:.3f}', ha='center', va='bottom', fontsize=9)  # Show the ratio above the bar
    plt.title("Color Histogram - " + img[0:4], fontsize=16)
    plt.xlabel("Color", fontsize=14)
    plt.ylabel("Ratio", fontsize=14)
    plt.savefig(os.path.join(folder, img[0:4] + "_color_histogram.png"))
    plt.close()

    # Gray Histogram
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.hist(img_gray.ravel(), bins=256, range=(0, 256), color='black', density=True)
    plt.title('Gray Histogram - ' + img[0:4], fontsize=16)
    plt.ylabel('Probability', fontsize=14)
    plt.xlabel('Pixel Value', fontsize=14)
    plt.savefig(os.path.join(folder, img[0:4] + '_gray_histogram.png'))
    plt.close()

    # Calculate gradient histogram using Sobel operator
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Plot gradient histogram
    plt.hist(grad_magnitude.ravel(), bins=361, range=(0, 360), color='black', density=True)
    plt.title('Gradient Histogram - ' + img[0:4], fontsize=16)
    plt.ylabel('Probability', fontsize=14)
    plt.xlabel('Gradient Magnitude', fontsize=14)
    plt.savefig(os.path.join(folder, img[0:4] + '_gradient_histogram.png'))
    plt.close()



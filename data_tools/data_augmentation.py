import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt


def random_brightness(img, max_delta=30):
    """
    Applies a random brightness adjustment to the input image.
"""
    
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(max(0, 1 - max_delta/100), 1 + max_delta/100)
    return enhancer.enhance(factor)


def random_color_jitter(img, max_delta=20):
    """
    Applies random color jitter to an RGB image for data augmentation.

    Each channel (R, G, B) is independently adjusted by adding a random value 
    uniformly sampled from [-max_delta, max_delta] to each pixel, and the result 
    is clipped to the [0, 255] range.

    """
    r, g, b = img.split()
    r = r.point(lambda i: np.clip(i + random.uniform(-max_delta, max_delta), 0, 255))
    g = g.point(lambda i: np.clip(i + random.uniform(-max_delta, max_delta), 0, 255))
    b = b.point(lambda i: np.clip(i + random.uniform(-max_delta, max_delta), 0, 255))
    return Image.merge('RGB', (r, g, b))

def random_blur(img, max_radius=1.5):
    """
    Applies a random Gaussian blur to the input image with a specified maximum radius.

    With a probability of 0.3, the function applies a Gaussian blur with a random radius between 0 and `max_radius`.
    Otherwise, the image is returned unchanged. The output image is always converted to RGB mode.
    """
    if random.random() > 0.7:
        radius = random.uniform(0, max_radius)
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return img.convert('RGB')

if __name__ == "__main__":
    # Example usage
    img = Image.open("/home/marta/Projects/tb/data/images/mch/1159/2/2023/01/01/1159_2_2023-01-01_1010.jpeg")  # Load an image
    img = random_brightness(img)
    img = random_color_jitter(img)
    img = random_blur(img)

    img.save("analysis/augmented/augmented_image.jpeg")  # Save the augmented image

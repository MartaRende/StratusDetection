import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter

def random_flip(img):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def random_rotate(img, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle)

def random_brightness(img, max_delta=30):
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(max(0, 1 - max_delta/100), 1 + max_delta/100)
    return enhancer.enhance(factor)

def random_contrast(img, max_delta=30):
    enhancer = ImageEnhance.Contrast(img)
    factor = random.uniform(max(0, 1 - max_delta/100), 1 + max_delta/100)
    return enhancer.enhance(factor)

def random_color_jitter(img, max_delta=20):
    r, g, b = img.split()
    r = r.point(lambda i: np.clip(i + random.uniform(-max_delta, max_delta), 0, 255))
    g = g.point(lambda i: np.clip(i + random.uniform(-max_delta, max_delta), 0, 255))
    b = b.point(lambda i: np.clip(i + random.uniform(-max_delta, max_delta), 0, 255))
    return Image.merge('RGB', (r, g, b))

def random_blur(img, max_radius=1.5):
    if random.random() > 0.7:
        radius = random.uniform(0, max_radius)
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return img


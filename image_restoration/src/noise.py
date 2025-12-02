import numpy as np
from skimage.util import random_noise
import scipy.ndimage as ndimage
from skimage.transform import resize

def add_gaussian_noise(image, mean=0, var=0.01):
    """
    Adds Gaussian noise to an image.
    """
    return random_noise(image, mode='gaussian', mean=mean, var=var)

def add_salt_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    """
    Adds Salt and Pepper noise to an image.
    """
    return random_noise(image, mode='s&p', amount=amount, salt_vs_pepper=salt_vs_pepper)

def add_speckle_noise(image, mean=0, var=0.05):
    """
    Adds Speckle noise to an image.
    """
    return random_noise(image, mode='speckle', mean=mean, var=var)

def add_periodic_noise(image, freq=0.1, amplitude=0.5):
    """
    Adds periodic (sinusoidal) noise to an image.
    Useful for demonstrating Fourier filtering.
    """
    row, col = image.shape
    x = np.arange(col)
    y = np.arange(row)
    X, Y = np.meshgrid(x, y)
    
    # Create sinusoidal noise
    noise = amplitude * np.sin(2 * np.pi * freq * X + 2 * np.pi * freq * Y)
    
    # Normalize image to 0-1 if it's not
    if image.max() > 1:
        image = image / 255.0
        
    noisy_image = image + noise
    
    # Clip to 0-1
    return np.clip(noisy_image, 0, 1)

def add_blur(image, sigma=2.0):
    """
    Adds Gaussian blur to the image.
    """
    return ndimage.gaussian_filter(image, sigma=sigma)

def add_pixelation(image, scale=0.2):
    """
    Simulates pixelation by downscaling and upscaling.
    """
    h, w = image.shape[:2]
    small_h, small_w = int(h * scale), int(w * scale)
    
    # Downscale
    small = resize(image, (small_h, small_w), anti_aliasing=True)
    
    # Upscale (Nearest Neighbor to keep pixelated look)
    pixelated = resize(small, (h, w), order=0, anti_aliasing=False)
    
    return pixelated

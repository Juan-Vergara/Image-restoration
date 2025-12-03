import numpy as np
import cv2

def analyze_image_noise(image):
    """
    Analyzes the image to detect specific noise patterns.
    Only detects: Periodic, Salt & Pepper, and Gaussian noise.
    
    Args:
        image: Input image (RGB or Grayscale).
        
    Returns:
        dict: A dictionary containing:
            - 'detected_noise': str
            - 'params': dict (parameters for restoration)
    """
    # Ensure image is float32 for OpenCV compatibility
    if image.dtype == np.float64:
        image = image.astype(np.float32)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # 1. Periodic Noise Detection (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    mean_mag = np.mean(magnitude_spectrum)
    std_mag = np.std(magnitude_spectrum)
    threshold = mean_mag + 4 * std_mag
    
    peaks = np.argwhere(magnitude_spectrum > threshold)
    notch_centers = []
    
    for r, c in peaks:
        dist_from_center = np.sqrt((r - crow)**2 + (c - ccol)**2)
        if dist_from_center < 20: 
            continue
        notch_centers.append((r, c))
        
    if len(notch_centers) > 5:
        return {
            "detected_noise": "Ruido Periódico",
            "params": {"notch_centers": notch_centers}
        }

    # 2. Salt & Pepper Detection
    n_pixels = gray.size
    n_zeros = np.sum(gray <= 0.01)
    n_ones = np.sum(gray >= 0.99)
    sp_ratio = (n_zeros + n_ones) / n_pixels
    
    if sp_ratio > 0.005:
        return {
            "detected_noise": "Sal y Pimienta",
            "params": {"amount": sp_ratio}
        }

    # 3. Gaussian Noise (MAD estimation)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    mad = np.median(np.abs(laplacian - np.median(laplacian)))
    sigma = 1.4826 * mad
    
    if sigma > 0.01:
        return {
            "detected_noise": "Ruido Gaussiano",
            "params": {"sigma": sigma}
        }
    
    # No noise detected
    return {
        "detected_noise": "Ninguno",
        "params": {}
    }

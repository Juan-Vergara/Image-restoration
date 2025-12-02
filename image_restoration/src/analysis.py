import numpy as np
import cv2

def analyze_image_noise(image):
    """
    Analyzes the image to detect specific noise patterns, particularly periodic noise.
    
    Args:
        image: Input image (RGB or Grayscale).
        
    Returns:
        dict: A dictionary containing:
            - 'detected_noise': str (e.g., 'Periodic', 'None')
            - 'confidence': float (0.0 to 1.0)
            - 'params': dict (parameters for restoration, e.g., 'notch_centers')
    """
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
        
    if len(notch_centers) > 5: # Lowered threshold for better detection
        return {
            "detected_noise": "Ruido Periódico",
            "confidence": 0.9,
            "params": {"notch_centers": notch_centers}
        }

    # 2. Salt & Pepper Detection
    n_pixels = gray.size
    n_zeros = np.sum(gray <= 0.01)
    n_ones = np.sum(gray >= 0.99)
    sp_ratio = (n_zeros + n_ones) / n_pixels
    
    if sp_ratio > 0.005: # Lowered threshold (was 0.01)
        return {
            "detected_noise": "Sal y Pimienta",
            "confidence": 0.8,
            "params": {"amount": sp_ratio}
        }

    # 3. Blur Detection (Variance of Laplacian)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 0.002: # Slightly increased threshold
        return {
            "detected_noise": "Desenfoque (Blur)",
            "confidence": 0.6,
            "params": {"sigma": 2.0}
        }

    # 4. Gaussian Noise Estimation
    # Improved estimation using high-frequency content
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Use MAD (Median Absolute Deviation) for robust estimation
    mad = np.median(np.abs(laplacian - np.median(laplacian)))
    sigma_est = 1.4826 * mad  # Convert MAD to sigma
    
    if sigma_est > 0.015: # Lowered threshold (was 0.02)
        return {
            "detected_noise": "Ruido Gaussiano",
            "confidence": 0.7,
            "params": {"sigma": sigma_est}
        }

    return {
        "detected_noise": "Ninguno Detectado",
        "confidence": 0.0,
        "params": {}
    }

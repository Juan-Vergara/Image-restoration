import numpy as np
import cv2

def apply_fft(image):
    """
    Applies FFT to an image and returns the shifted frequency spectrum.
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return fshift

def apply_ifft(fshift):
    """
    Applies Inverse FFT to a shifted spectrum and returns the spatial image.
    """
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

def ideal_lowpass(shape, cutoff):
    """
    Creates an Ideal Low Pass Filter mask.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    
    # Create a circle mask
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
    mask[mask_area] = 1
    return mask

def butterworth_lowpass(shape, cutoff, order=2):
    """
    Creates a Butterworth Low Pass Filter mask.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    y, x = np.ogrid[:rows, :cols]
    radius = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Avoid division by zero
    radius = np.maximum(radius, 1e-5)
    
    mask = 1 / (1 + (radius / cutoff)**(2 * order))
    return mask

def notch_filter(shape, centers, radius=10, order=2):
    """
    Creates a Butterworth Notch Filter mask to reject specific frequencies.
    OPTIMIZED: Pre-computes distance grids for speed.
    """
    rows, cols = shape
    mask = np.ones((rows, cols), np.float32)
    
    # Pre-compute coordinate grids (OPTIMIZATION)
    y, x = np.ogrid[:rows, :cols]
    
    for center in centers:
        crow, ccol = center
        
        # Compute distance once (OPTIMIZED)
        dist_sq = (x - ccol)**2 + (y - crow)**2
        dist = np.sqrt(dist_sq)
        
        # Butterworth notch (smooth transition)
        notch_response = 1 / (1 + (radius / (dist + 1e-5))**(2 * order))
        mask *= notch_response
        
        # Symmetric point (OPTIMIZED: reuse computation)
        sym_row, sym_col = rows - crow, cols - ccol
        dist_sym_sq = (x - sym_col)**2 + (y - sym_row)**2
        dist_sym = np.sqrt(dist_sym_sq)
        notch_response_sym = 1 / (1 + (radius / (dist_sym + 1e-5))**(2 * order))
        mask *= notch_response_sym
        
    return mask

def apply_filter(image, filter_type='ideal', cutoff=30, order=2, notch_centers=None):
    """
    Applies a frequency domain filter to an image.
    """
    fshift = apply_fft(image)
    
    if filter_type == 'ideal':
        mask = ideal_lowpass(image.shape, cutoff)
    elif filter_type == 'butterworth':
        mask = butterworth_lowpass(image.shape, cutoff, order)
    elif filter_type == 'notch':
        if notch_centers is None:
             # Default to center if no centers provided (essentially blocks DC)
             # But usually notch is for specific noise. 
             # For generic usage without detection, this might not be useful.
             # We'll return original if no centers.
             return image, fshift, fshift
        
        # If notch_centers are provided from analysis, they are exact points.
        # We need to ensure the mask function handles them correctly.
        # The current notch_filter function takes a list of centers and a radius.
        mask = notch_filter(image.shape, notch_centers, radius=cutoff)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
        
    fshift_filtered = fshift * mask
    img_back = apply_ifft(fshift_filtered)
    
    return img_back, fshift, fshift_filtered

def wiener_deconvolution(image, psf, balance=0.1):
    """
    Performs Wiener Deconvolution in the frequency domain.
    image: Blurred/Noisy image.
    psf: Point Spread Function (Kernel) that caused the blur.
    balance: Regularization parameter (approx inverse of SNR).
    """
    # 1. FFT of Image
    img_fft = np.fft.fft2(image)
    
    # 2. FFT of PSF (Pad PSF to image size)
    psf_padded = np.zeros_like(image)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf
    
    # Center the PSF
    psf_padded = np.roll(psf_padded, -kh//2, axis=0)
    psf_padded = np.roll(psf_padded, -kw//2, axis=1)
    
    psf_fft = np.fft.fft2(psf_padded)
    
    # 3. Wiener Filter Formula: G = (H* / (|H|^2 + K)) * F_noisy
    # H is psf_fft
    psf_fft_conj = np.conj(psf_fft)
    
    # Avoid division by zero
    denominator = (np.abs(psf_fft)**2 + balance)
    
    result_fft = (psf_fft_conj / denominator) * img_fft
    
    # 4. Inverse FFT
    result = np.abs(np.fft.ifft2(result_fft))
    
    return result, np.fft.fftshift(result_fft), np.fft.fftshift(img_fft)

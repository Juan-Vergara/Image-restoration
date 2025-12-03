import numpy as np
import cv2
import scipy.ndimage as ndimage
from skimage.restoration import (denoise_tv_chambolle, denoise_wavelet, 
                                 denoise_nl_means, estimate_sigma, denoise_bilateral, 
                                 richardson_lucy, denoise_tv_bregman)
from skimage.filters import median, unsharp_mask
from skimage.morphology import disk
from scipy.signal import wiener

try:
    import bm3d
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False
    print("Warning: BM3D not available. Install with: pip install bm3d")

def apply_gaussian(image, sigma=1):
    return ndimage.gaussian_filter(image, sigma=sigma)

def apply_median(image, disk_radius=2):
    """
    Applies median filter - excellent for salt & pepper noise.
    disk_radius: Size of the structuring element (1-5)
    """
    # Ensure image is in valid range
    image = np.clip(image, 0, 1)
    
    if len(image.shape) == 3:
        # Color image: process each channel independently
        channels = [image[:,:,i] for i in range(3)]
        restored_channels = []
        for ch in channels:
            restored_channels.append(median(ch, disk(disk_radius)))
        result = np.stack(restored_channels, axis=2)
    else:
        # Grayscale
        result = median(image, disk(disk_radius))
        
    return np.clip(result, 0, 1)

def apply_wiener(image, window_size=5):
    # Wiener filter in scipy.signal
    return wiener(image, (window_size, window_size))

def apply_wavelet(image):
    # BayesShrink is the default method in skimage denoise_wavelet when method='BayesShrink'
    # But 'BayesShrink' is a specific thresholding rule. 
    # We'll use default soft thresholding with BayesShrink estimation implicitly or explicitly.
    # skimage's denoise_wavelet uses BayesShrink by default for sigma estimation if not provided?
    # Actually, let's use a standard call.
    return denoise_wavelet(image, method='BayesShrink', mode='soft', rescale_sigma=True)

def apply_tv(image, weight=0.1):
    return denoise_tv_chambolle(image, weight=weight)

def apply_nlm(image):
    # Estimate sigma for NLM
    sigma_est = np.mean(estimate_sigma(image))
    
    # NLM is slow, so we use fast approximation if possible or small patch size
    return denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True,
                            patch_size=5, patch_distance=6)

def apply_bilateral(image, sigma_color=0.1, sigma_spatial=5):
    """
    Applies Bilateral filter.
    sigma_color: Standard deviation for intensity difference (0.05-0.2)
    sigma_spatial: Standard deviation for spatial distance (3-15)
    """
    # Ensure image is in valid range
    image = np.clip(image, 0, 1)
    result = denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=None)
    return np.clip(result, 0, 1)

def apply_nlm_opencv(image, h=3, templateWindowSize=7, searchWindowSize=21):
    """
    Applies Non-Local Means Denoising using OpenCV (Faster).
    Image expected to be float [0,1].
    h: Filter strength (3-15). Lower = preserve detail, Higher = more smoothing
    """
    # Convert to uint8 for OpenCV
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Apply NLM
    dst = cv2.fastNlMeansDenoising(img_uint8, None, h=h, 
                                    templateWindowSize=templateWindowSize, 
                                    searchWindowSize=searchWindowSize)
    
    return np.clip(dst.astype(np.float32) / 255.0, 0, 1)

def apply_unsharp_mask(image, radius=1.0, amount=1.0):
    """
    Applies Unsharp Masking to sharpen the image.
    """
    return unsharp_mask(image, radius=radius, amount=amount)

def apply_richardson_lucy(image, psf, iterations=30):
    """
    Applies Richardson-Lucy deconvolution.
    """
    # RL expects float 0-1
    return richardson_lucy(image, psf, num_iter=iterations)

def apply_superres_lanczos(image, scale=2.0):
    """
    Simulates Super Resolution using Lanczos upscaling followed by sharpening.
    This is a 'fake' super res but effective for simple demo.
    """
    # Upscale
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    
    # cv2.INTER_LANCZOS4 is high quality
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Sharpen
    restored = unsharp_mask(upscaled, radius=1.0, amount=1.5)
    
    return restored

def apply_ai_denoise(image, model_path="dncnn.onnx"):
    """
    Applies AI Denoising using OpenCV DNN module.
    Expects an ONNX model.
    """
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
    except Exception:
        # Fallback if model not found
        return None

    # Prepare blob
    # DnCNN usually expects (1, 1, H, W) for grayscale
    img_float = image.astype(np.float32)
    blob = cv2.dnn.blobFromImage(img_float, scalefactor=1.0, size=None, mean=0, swapRB=False, crop=False)
    
    net.setInput(blob)
    output = net.forward()
    
    # Post-process
    result = output[0, 0, :, :]
    return np.clip(result, 0, 1)

def apply_bm3d(image, sigma_psd=0.02):
    """
    Applies BM3D denoising - State of the art for Gaussian noise.
    sigma_psd: Noise standard deviation (0.01-0.1)
    """
    if not BM3D_AVAILABLE:
        # Fallback to bilateral if BM3D not available
        print("BM3D not available, using Bilateral filter as fallback")
        return apply_bilateral(image, sigma_color=0.1, sigma_spatial=5)
    
    # Ensure image is in valid range
    image = np.clip(image, 0, 1)
    
    try:
        # BM3D expects noise std in [0,1] range
        denoised = bm3d.bm3d(image, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        return np.clip(denoised, 0, 1)
    except Exception as e:
        print(f"BM3D failed: {e}, using fallback")
        return apply_bilateral(image, sigma_color=0.1, sigma_spatial=5)

def apply_tv_bregman(image, weight=0.1, max_iter=100, eps=0.001):
    """
    Applies Total Variation denoising with Bregman iteration.
    Better edge preservation than standard TV.
    """
    image = np.clip(image, 0, 1)
    result = denoise_tv_bregman(image, weight=weight, max_num_iter=max_iter, eps=eps)
    return np.clip(result, 0, 1)

def apply_wiener_adaptive(image, noise_variance=None):
    """
    Applies adaptive Wiener filter with automatic noise estimation.
    """
    image = np.clip(image, 0, 1)
    
    if noise_variance is None:
        # Estimate noise variance from image
        noise_variance = estimate_sigma(image, average_sigmas=True) ** 2
    
    # Apply Wiener filter
    result = wiener(image, (5, 5), noise=noise_variance)
    return np.clip(result, 0, 1)

def apply_adaptive_median(image, max_window_size=7):
    """
    Adaptive Median Filter - SPECIALIZED for Salt & Pepper noise.
    Better than standard median as it adapts window size.
    """
    image = np.clip(image, 0, 1)
    
    # Convert to uint8 for processing
    img_uint8 = (image * 255).astype(np.uint8)
    
    if len(img_uint8.shape) == 3:
        # Color image: process each channel independently
        channels = cv2.split(img_uint8)
        restored_channels = []
        for ch in channels:
            restored_channels.append(_adaptive_median_2d(ch, max_window_size))
        result = cv2.merge(restored_channels)
    else:
        # Grayscale
        result = _adaptive_median_2d(img_uint8, max_window_size)
    
    return np.clip(result.astype(np.float32) / 255.0, 0, 1)

def _adaptive_median_2d(img, max_window_size):
    rows, cols = img.shape
    result = img.copy()
    
    for i in range(rows):
        for j in range(cols):
            window_size = 3
            while window_size <= max_window_size:
                # Get window
                half = window_size // 2
                i_min = max(0, i - half)
                i_max = min(rows, i + half + 1)
                j_min = max(0, j - half)
                j_max = min(cols, j + half + 1)
                
                window = img[i_min:i_max, j_min:j_max]
                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)
                z_xy = img[i, j]
                
                # Level A
                if z_min < z_med < z_max:
                    # Level B
                    if z_min < z_xy < z_max:
                        result[i, j] = z_xy
                    else:
                        result[i, j] = z_med
                    break
                else:
                    window_size += 2
                    
                if window_size > max_window_size:
                    result[i, j] = z_med
    return result

def apply_lee_filter(image, window_size=5):
    """
    Lee Filter - SPECIALIZED for Speckle noise (multiplicative).
    Used in SAR and ultrasound imaging.
    """
    image = np.clip(image, 0, 1)
    
    # Compute local statistics
    from scipy.ndimage import uniform_filter
    
    # Local mean
    mean = uniform_filter(image, size=window_size)
    
    # Local variance
    sqr_mean = uniform_filter(image**2, size=window_size)
    variance = sqr_mean - mean**2
    
    # Overall variance (noise variance estimate)
    overall_variance = np.var(image)
    
    # Weighting factor
    weights = variance / (variance + overall_variance + 1e-10)
    
    # Lee filter formula
    result = mean + weights * (image - mean)
    
    return np.clip(result, 0, 1)

def apply_richardson_lucy_optimized(image, psf=None, iterations=10):
    """
    Optimized Richardson-Lucy deconvolution for blur removal.
    SPECIALIZED for defocus/motion blur.
    """
    image = np.clip(image, 0, 1)
    
    if psf is None:
        # Create default Gaussian PSF
        from scipy.signal import gaussian
        size = 9
        sigma = 2.0
        ax = gaussian(size, sigma)
        psf = np.outer(ax, ax)
        psf = psf / psf.sum()
    
    # Use fewer iterations for speed (10 instead of 30)
    result = richardson_lucy(image, psf, num_iter=iterations, clip=False)
    return np.clip(result, 0, 1)


from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original, restored):
    """
    Calculates Peak Signal-to-Noise Ratio.
    """
    # Ensure data ranges match. If original is 0-1 and restored is 0-255, fix it.
    # We assume our pipeline uses 0-1 floats mostly.
    return psnr(original, restored, data_range=original.max() - original.min())

def calculate_ssim(original, restored):
    """
    Calculates Structural Similarity Index.
    """
    # SSIM requires specifying data_range for float images
    return ssim(original, restored, data_range=original.max() - original.min())

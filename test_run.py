import numpy as np
from image_restoration.src.noise import add_gaussian_noise, add_periodic_noise
from image_restoration.src.fourier import apply_filter
from image_restoration.src.spatial import apply_gaussian, apply_median
from image_restoration.src.metrics import calculate_psnr, calculate_ssim

def test_pipeline():
    print("Creating dummy image...")
    img = np.zeros((256, 256))
    img[64:192, 64:192] = 1.0
    
    print("Adding noise...")
    noisy = add_gaussian_noise(img)
    noisy_periodic = add_periodic_noise(img)
    
    print("Testing Fourier Filter...")
    restored_fft, _, _ = apply_filter(noisy_periodic, filter_type='notch', cutoff=10)
    
    print("Testing Spatial Filter...")
    restored_spatial = apply_gaussian(noisy)
    
    print("Calculating Metrics...")
    psnr = calculate_psnr(img, restored_spatial)
    ssim = calculate_ssim(img, restored_spatial)
    
    print(f"PSNR: {psnr}, SSIM: {ssim}")
    print("Test passed!")

if __name__ == "__main__":
    test_pipeline()

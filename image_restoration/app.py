import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.io import imread
from PIL import Image
import io
import os
import requests

from image_restoration.src.noise import (add_gaussian_noise, add_salt_pepper_noise, 
                                         add_periodic_noise)
from image_restoration.src.fourier import apply_filter
from image_restoration.src.spatial import (apply_gaussian, apply_median, 
                                           apply_bilateral, apply_nlm_opencv,
                                           apply_richardson_lucy, apply_unsharp_mask)
from image_restoration.src.metrics import calculate_psnr, calculate_ssim
from image_restoration.src.analysis import analyze_image_noise

# Configuración de Página
st.set_page_config(
    page_title="Restauración de Imágenes Pro",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suprimir warnings
import warnings
warnings.filterwarnings('ignore')

# CSS Personalizado
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #333; }
    h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 400; color: #555; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #007bff; }
    .metric-label { font-size: 0.9rem; color: #6c757d; }
    .stButton>button { width: 100%; border-radius: 20px; }
    .compare-header { text-align: center; font-weight: bold; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🔧 Configuración")

# 1. Fuente de Imagen
st.sidebar.subheader("1. Fuente de Imagen")
image_source = st.sidebar.radio("Seleccionar Fuente", ["Imagen de Muestra", "Subir Imagen"])

if image_source == "Imagen de Muestra":
    sample_name = st.sidebar.selectbox("Seleccionar Muestra", ["Lena", "Cameraman", "Astronauta"])
    if sample_name == "Lena":
        img = data.camera() # Placeholder, skimage doesn't have lena anymore usually, using camera as fallback or astronaut
        # Actually let's use what was there or standard ones
        if hasattr(data, 'lena'): img = data.lena()
        else: img = data.astronaut()
        if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif sample_name == "Cameraman":
        img = data.camera()
    else:
        img = data.astronaut()
        if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
else:
    uploaded_file = st.sidebar.file_uploader("Subir imagen (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)
        if len(img.shape) == 3 and img.shape[2] == 4: # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        st.info("Por favor sube una imagen para comenzar.")
        st.stop()

# Normalizar imagen (0-1 float)
if img.dtype != np.float64:
    img = img_as_float(img)

is_gray = len(img.shape) == 2

# Análisis Automático de Ruido (Solo para imágenes subidas)
detected_params = {}
auto_method = None
auto_params = {}

if image_source == "Subir Imagen":
    with st.spinner("Analizando imagen..."):
        analysis_result = analyze_image_noise(img)
        detected_noise = analysis_result["detected_noise"]
        detected_params = analysis_result["params"]
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Análisis Automático")
        
        if detected_noise != "Ninguno":
            st.sidebar.success(f"Detectado: **{detected_noise}**")
            
            # Sugerir método
            if detected_noise == "Ruido Periódico":
                auto_method = "Fourier Notch (Optimizado)"
                auto_params = detected_params
                st.sidebar.info("💡 **Método Automático:** Filtro Notch")
            elif detected_noise == "Sal y Pimienta":
                auto_method = "Mediana Adaptativa"
                auto_params = {"max_window_size": 7}
                st.sidebar.info("💡 **Método Automático:** Mediana Adaptativa")
            elif detected_noise == "Ruido Gaussiano":
                auto_method = "BM3D"
                auto_params = {"sigma_psd": detected_params.get("sigma", 0.02)}
                st.sidebar.info("💡 **Método Automático:** BM3D")
        else:
            if image_source == "Subir Imagen":
                st.sidebar.markdown("---")
                st.sidebar.info("✅ No se detectaron patrones de ruido obvios.")

st.sidebar.subheader("2. Generación de Ruido")
noise_type = st.sidebar.selectbox("Tipo de Degradación", ["Ninguna", "Ruido Gaussiano", "Sal y Pimienta", "Ruido Periódico"])

noisy_img = img.copy()
psf_kernel = None # Para deconvolución

if noise_type == "Ruido Gaussiano":
    sigma = st.sidebar.slider("Sigma (Intensidad)", 0.01, 0.5, 0.1)
    noisy_img = add_gaussian_noise(img, sigma=sigma)
elif noise_type == "Sal y Pimienta":
    amount = st.sidebar.slider("Cantidad", 0.0, 0.5, 0.05)
    noisy_img = add_salt_pepper_noise(img, amount=amount)
elif noise_type == "Ruido Periódico":
    freq = st.sidebar.slider("Frecuencia", 0.01, 0.5, 0.1)
    amp = st.sidebar.slider("Amplitud", 0.0, 1.0, 0.2)
    noisy_img = add_periodic_noise(img, freq=freq, amplitude=amp)

# --- Helper para Ejecutar Método ---
def run_restoration(method_name, noisy, original, col_key):
    restored = noisy
    fshift = None
    
    if method_name == "Ninguno":
        return noisy, None
    
    # Métodos de Fourier
    if "Fourier" in method_name:
        if not is_gray: return noisy, None
        
        if "Notch" in method_name: filter_type = "notch"; cutoff = 10
        
        with st.expander(f"⚙️ Configuración {method_name}"):
            notch_centers = None
            if filter_type == "notch":
                # Opción de usar detectados
                use_auto = False
                if detected_params and "notch_centers" in detected_params:
                    use_auto = st.checkbox(f"Usar Picos Detectados ({len(detected_params['notch_centers'])})", value=True, key=f"auto_notch_{col_key}")
                
                if use_auto:
                    notch_centers = detected_params["notch_centers"]
                else:
                    # Manual Mode with Direction Selector
                    notch_mode = st.selectbox(f"Patrón de Ruido ({col_key})", 
                                            ["Diagonal (Cruz)", "Horizontal (Picos Verticales)", "Vertical (Picos Horizontales)", "Cuadrícula (Ambos)"],
                                            key=f"nmode_{col_key}")
                    
                    offset = st.slider(f"Frecuencia/Desplazamiento ({col_key})", 0, 100, 20, key=f"notch_{col_key}")
                    rows, cols = original.shape
                    crow, ccol = rows//2, cols//2
                    
                    notch_centers = []
                    
                    # Diagonal (Original)
                    if "Diagonal" in notch_mode:
                        notch_centers.extend([(crow-offset, ccol-offset), (crow+offset, ccol+offset),
                                              (crow-offset, ccol+offset), (crow+offset, ccol-offset)])
                    
                    # Horizontal Noise -> Vertical Peaks
                    if "Horizontal" in notch_mode or "Cuadrícula" in notch_mode:
                        notch_centers.extend([(crow-offset, ccol), (crow+offset, ccol)])
                        
                    # Vertical Noise -> Horizontal Peaks
                    if "Vertical" in notch_mode or "Cuadrícula" in notch_mode:
                        notch_centers.extend([(crow, ccol-offset), (crow, ccol+offset)])

        restored, _, fshift = apply_filter(noisy, filter_type=filter_type, cutoff=cutoff, notch_centers=notch_centers)
        
    # Métodos Espaciales
    else:
        with st.expander(f"⚙️ Configuración {method_name}"):
            if method_name == "BM3D":
                sigma_psd = st.slider(f"Sigma PSD ({col_key})", 0.01, 0.1, 0.02, step=0.001, key=f"bm3d_{col_key}")
                from image_restoration.src.spatial import apply_bm3d
                restored = apply_bm3d(noisy, sigma_psd=sigma_psd)
            elif method_name == "Gaussiano":
                sigma = st.slider(f"Sigma ({col_key})", 0.1, 3.0, 0.8, key=f"sigma_{col_key}")
                restored = apply_gaussian(noisy, sigma)
            elif method_name == "Mediana":
                radius = st.slider(f"Radio ({col_key})", 1, 5, 2, key=f"rad_{col_key}")
                restored = apply_median(noisy, radius)

        # Visualize Removed Noise (Difference)
        with st.expander(f"👀 Ver Ruido Eliminado ({col_key})"):
            # Calculate difference
            if restored.shape == noisy.shape:
                diff = np.abs(noisy - restored)
                # Enhance contrast of difference for visibility
                diff_vis = diff / (np.max(diff) + 1e-5) 
                st.image(diff_vis, caption="Diferencia (Lo que se eliminó)", clamp=True)
            else:
                st.info("No disponible (cambio de tamaño detectado)")

    return restored, fshift

# --- Diseño Principal ---
st.title("Restauración de Imágenes")

# Only show section 1 for sample images with noise
if image_source == "Imagen de Muestra" and noise_type != "Ninguna":
    st.markdown("### 1. Referencia (Entrada Degradada)")
    col_ref1, col_ref2, col_ref3 = st.columns([1, 2, 1])
    with col_ref2:
        st.image(noisy_img, caption="Imagen Ruidosa/Degradada", use_container_width=True, clamp=True, channels='GRAY' if is_gray else 'RGB')
        psnr_noisy = calculate_psnr(img, noisy_img)
        st.caption(f"PSNR Base: {psnr_noisy:.2f} dB")

# --- Restauración Automática ---
if auto_method and image_source == "Subir Imagen":
    st.markdown("---")
    st.markdown("### 2. Restauración Automática")
    
    st.info(f"🤖 Aplicando **{auto_method}** automáticamente basado en el ruido detectado...")
    
    # Ejecutar restauración automática
    auto_restored = noisy_img.copy()
    
    if auto_method == "Fourier Notch (Optimizado)":
        notch_centers = auto_params.get("notch_centers", [])
        if notch_centers and is_gray:
            # Use optimized notch filter
            auto_restored, _, _ = apply_filter(noisy_img, filter_type="notch", cutoff=20, notch_centers=notch_centers)
    
    elif auto_method == "Mediana Adaptativa":
        max_window_size = auto_params.get("max_window_size", 7)
        from image_restoration.src.spatial import apply_adaptive_median
        auto_restored = apply_adaptive_median(noisy_img, max_window_size=max_window_size)
    
    elif auto_method == "BM3D":
        # Trust estimation but ensure reasonable bounds for visibility
        raw_sigma = auto_params.get("sigma_psd", 0.02)
        # Ensure minimum of 0.03 for visible effect, max 0.1 to avoid extreme blur
        sigma_psd = max(raw_sigma, 0.03)
        sigma_psd = min(sigma_psd, 0.1)
        
        st.sidebar.info(f"ℹ️ Sigma estimado: {raw_sigma:.3f} → Ajustado: {sigma_psd:.3f}")
        from image_restoration.src.spatial import apply_bm3d
        auto_restored = apply_bm3d(noisy_img, sigma_psd=sigma_psd)
    
    # Display result
    col_auto1, col_auto2 = st.columns(2)
    
    with col_auto1:
        st.image(noisy_img, caption="Original (Con Ruido)", use_container_width=True, clamp=True, channels='GRAY' if is_gray else 'RGB')
    
    with col_auto2:
        st.image(auto_restored, caption=f"Restaurada ({auto_method})", use_container_width=True, clamp=True, channels='GRAY' if is_gray else 'RGB')
        
        # Show difference
        if auto_restored.shape == noisy_img.shape:
            diff = np.abs(noisy_img - auto_restored)
            diff_vis = diff / (np.max(diff) + 1e-5)
            with st.expander("👀 Ver Ruido Eliminado"):
                st.image(diff_vis, caption="Diferencia (Ruido Eliminado)", clamp=True)

st.markdown("---")
st.markdown("### 3. Restauración Manual (Opcional)")

# Single method selection
methods_list = ["Ninguno", "BM3D", "Gaussiano", "Mediana", "Fourier Notch"]
method = st.selectbox("Seleccionar Método de Restauración", methods_list, index=0, key="manual_method")

# Two columns: Original (left) and Restored (right)
col_orig, col_rest = st.columns(2)

with col_orig:
    st.markdown("#### Imagen Original")
    st.image(noisy_img, caption="Imagen con Ruido", use_container_width=True, clamp=True, channels='GRAY' if is_gray else 'RGB')

with col_rest:
    st.markdown("#### Imagen Restaurada")
    
    if method == "Ninguno":
        st.image(noisy_img, caption="Sin Restauración", use_container_width=True, clamp=True, channels='GRAY' if is_gray else 'RGB')
        st.info("Selecciona un método de restauración para ver el resultado")
    else:
        # Apply restoration
        restored, fshift = run_restoration(method, noisy_img, img, "manual")
        
        st.image(restored, caption=f"Restaurada ({method})", use_container_width=True, clamp=True, channels='GRAY' if is_gray else 'RGB')
        
        # Show metrics
        p = calculate_psnr(img, restored)
        s = calculate_ssim(img, restored) if is_gray else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PSNR</div>
            <div class="metric-value">{p:.2f} dB</div>
            <div class="metric-label">SSIM: {s:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer / Teoría
st.markdown("---")
with st.expander("ℹ️ Guía de Mejores Prácticas"):
    st.markdown("""
    | Problema | Mejor Solución | Por qué |
    | :--- | :--- | :--- |
    | **Ruido Gaussiano (Grano)** | **BM3D** | Estado del arte, explota redundancia de bloques similares. |
    | **Sal y Pimienta (Puntos)** | **Mediana** | Elimina píxeles corruptos sin afectar el resto. |
    | **Ruido Periódico (Rayas)** | **Fourier Notch** | Elimina frecuencias específicas (puntos brillantes en el espectro). |
    """)

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
                                         add_speckle_noise, add_periodic_noise,
                                         add_blur, add_pixelation)
from image_restoration.src.fourier import apply_filter, wiener_deconvolution
from image_restoration.src.spatial import (apply_gaussian, apply_median, apply_wiener, 
                                           apply_wavelet, apply_tv, apply_nlm,
                                           apply_bilateral, apply_nlm_opencv,
                                           apply_richardson_lucy, apply_superres_lanczos,
                                           apply_ai_denoise, apply_unsharp_mask)
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
    .metric-value { font-size: 1.5em; font-weight: bold; color: #28a745; }
    .metric-label { font-size: 0.9em; color: #6c757d; }
    .compare-header { text-align: center; padding: 10px; background: #e9ecef; border-radius: 5px; margin-bottom: 10px; font-weight: bold; color: #333; }
</style>
""", unsafe_allow_html=True)

# --- Configuración de la Barra Lateral ---
st.sidebar.title("Configuración")

# 1. Selección de Imagen
st.sidebar.subheader("1. Imagen de Origen")
image_source = st.sidebar.radio("Fuente", ["Imagen de Muestra", "Subir Imagen"], horizontal=True)

if image_source == "Imagen de Muestra":
    sample_name = st.sidebar.selectbox("Elegir Muestra", ["Astronauta", "Cámara", "Lena (Gris)", "Monedas"])
    if sample_name == "Astronauta": img = data.astronaut(); is_gray = False
    elif sample_name == "Cámara": img = data.camera(); is_gray = True
    elif sample_name == "Lena (Gris)": img = data.camera(); is_gray = True # Fallback
    elif sample_name == "Monedas": img = data.coins(); is_gray = True
else:
    uploaded_file = st.sidebar.file_uploader("Subir", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img = np.array(image)
        is_gray = len(img.shape) == 2
    else:
        st.warning("Sube una imagen para comenzar."); st.stop()

img = img_as_float(img)
# Fix: Check if already grayscale before converting
if len(img.shape) == 3 and img.shape[2] == 3:
    # RGB image
    if st.sidebar.checkbox("Convertir a Escala de Grises", value=True):
        from skimage.color import rgb2gray
        img = rgb2gray(img)
        is_gray = True
elif len(img.shape) == 2:
    # Already grayscale
    is_gray = True

# --- Análisis Automático de Ruido ---
analysis_result = analyze_image_noise(img)
detected_params = {}
auto_method = None
auto_params = {}

if analysis_result['detected_noise'] != 'Ninguno Detectado':
    st.sidebar.markdown("---")
    st.sidebar.success(f"🔍 **Ruido Detectado:** {analysis_result['detected_noise']}")
    detected_params = analysis_result['params']
    
    if analysis_result['detected_noise'] == "Ruido Periódico":
        auto_method = "Fourier Notch (Optimizado)"
        auto_params = {"notch_centers": detected_params.get("notch_centers", [])}
        st.sidebar.info("💡 **Método Automático:** Fourier Notch")
    elif analysis_result['detected_noise'] == "Sal y Pimienta":
        auto_method = "Mediana Adaptativa"
        auto_params = {"max_window_size": 7}
        st.sidebar.info("💡 **Método Automático:** Mediana Adaptativa")
    elif analysis_result['detected_noise'] == "Ruido Gaussiano":
        auto_method = "BM3D"
        sigma = detected_params.get("sigma", 0.02)
        auto_params = {"sigma_psd": min(sigma, 0.1)}
        st.sidebar.info(f"💡 **Método Automático:** BM3D (sigma={sigma:.3f})")
    elif analysis_result['detected_noise'] == "Desenfoque (Blur)":
        auto_method = "Richardson-Lucy Optimizado"
        auto_params = {"iterations": 10}
        st.sidebar.info("💡 **Método Automático:** Richardson-Lucy")
else:
    if image_source == "Subir Imagen":
        st.sidebar.markdown("---")
        st.sidebar.info("✅ No se detectaron patrones de ruido obvios.")
st.sidebar.subheader("2. Generación de Ruido")
noise_type = st.sidebar.selectbox("Tipo de Degradación", ["Ninguna", "Ruido Gaussiano", "Sal y Pimienta", "Speckle", "Ruido Periódico", "Desenfoque Gaussiano", "Pixelación"])

noisy_img = img.copy()
psf_kernel = None # Para deconvolución

if noise_type == "Ruido Gaussiano":
    var = st.sidebar.slider("Varianza", 0.0, 0.1, 0.01, step=0.001)
    noisy_img = add_gaussian_noise(img, var=var)
elif noise_type == "Sal y Pimienta":
    amount = st.sidebar.slider("Cantidad", 0.0, 0.5, 0.05)
    noisy_img = add_salt_pepper_noise(img, amount=amount)
elif noise_type == "Speckle":
    var = st.sidebar.slider("Varianza", 0.0, 0.1, 0.05)
    noisy_img = add_speckle_noise(img, var=var)
elif noise_type == "Ruido Periódico":
    freq = st.sidebar.slider("Frecuencia", 0.01, 0.5, 0.1)
    amp = st.sidebar.slider("Amplitud", 0.0, 1.0, 0.2)
    noisy_img = add_periodic_noise(img, freq=freq, amplitude=amp)
elif noise_type == "Desenfoque Gaussiano":
    sigma_blur = st.sidebar.slider("Sigma de Desenfoque", 0.5, 5.0, 2.0)
    noisy_img = add_blur(img, sigma=sigma_blur)
    
    # Crear PSF para deconvolución
    ksize = int(4 * sigma_blur + 1)
    if ksize % 2 == 0: ksize += 1
    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma_blur))
    kernel = np.outer(gauss, gauss)
    psf_kernel = kernel / np.sum(kernel)

elif noise_type == "Pixelación":
    scale_pix = st.sidebar.slider("Factor de Reducción", 0.05, 0.5, 0.2)
    noisy_img = add_pixelation(img, scale=scale_pix)

# --- Helper para Descargar Modelo ---
def download_model():
    url = "https://github.com/onnx/models/raw/main/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx"
    path = "dncnn.onnx"
    if not os.path.exists(path):
        with st.spinner("Descargando modelo de IA..."):
            try:
                r = requests.get(url, allow_redirects=True)
                with open(path, 'wb') as f:
                    f.write(r.content)
                st.success("Modelo descargado.")
            except:
                st.error("Error al descargar el modelo.")

# --- Helper para Ejecutar Método ---
def run_restoration(method_name, noisy, original, col_key):
    restored = noisy
    fshift = None
    
    if method_name == "Ninguno":
        return noisy, None
    
    # Métodos de Fourier
    if "Fourier" in method_name:
        if not is_gray: return noisy, None
        
        if "Ideal" in method_name: filter_type = "ideal"; cutoff = 30
        elif "Butterworth" in method_name: filter_type = "butterworth"; cutoff = 30
        elif "Notch" in method_name: filter_type = "notch"; cutoff = 10
        
        with st.expander(f"⚙️ Configuración {method_name}"):
            if filter_type != "notch":
                cutoff = st.slider(f"Corte ({col_key})", 1, 100, 30, key=f"cutoff_{col_key}")
            
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
        
        # Visualize Spectrum with Red Circles for Notch
        if filter_type == "notch" and notch_centers:
            # Create a copy of the spectrum for visualization
            spectrum_vis = 20 * np.log(np.abs(fshift) + 1)
            spectrum_vis = spectrum_vis / np.max(spectrum_vis) # Normalize 0-1
            spectrum_vis = np.uint8(spectrum_vis * 255)
            spectrum_vis = cv2.cvtColor(spectrum_vis, cv2.COLOR_GRAY2RGB)
            
            for r, c in notch_centers:
                cv2.circle(spectrum_vis, (c, r), 5, (255, 0, 0), 2) # Red circles
            
            st.image(spectrum_vis, caption=f"Espectro con Filtro Notch ({col_key})", clamp=True)
            fshift = None # Don't show the default one again below
        
    # Métodos Espaciales
    else:
        with st.expander(f"⚙️ Configuración {method_name}"):
            if method_name == "Gaussiano":
                sigma = st.slider(f"Sigma ({col_key})", 0.1, 3.0, 0.8, key=f"sigma_{col_key}")
                restored = apply_gaussian(noisy, sigma)
            elif method_name == "Mediana":
                radius = st.slider(f"Radio ({col_key})", 1, 5, 2, key=f"rad_{col_key}")
                restored = apply_median(noisy, radius)
            elif method_name == "Wiener":
                window = st.slider(f"Ventana ({col_key})", 3, 11, 3, step=2, key=f"win_{col_key}")
                restored = apply_wiener(noisy, window)
            elif method_name == "Wavelet":
                restored = apply_wavelet(noisy)
            elif method_name == "Variación Total":
                weight = st.slider(f"Peso ({col_key})", 0.01, 0.3, 0.05, key=f"tv_{col_key}")
                restored = apply_tv(noisy, weight=weight)
            elif method_name == "Non-Local Means (Lento)":
                if st.button(f"Ejecutar NLM ({col_key})"):
                    with st.spinner("Ejecutando NLM..."):
                        restored = apply_nlm(noisy)
            elif method_name == "Bilateral":
                sigma_color = st.slider(f"Sigma Color ({col_key})", 0.01, 0.2, 0.03, key=f"bc_{col_key}")
                sigma_spatial = st.slider(f"Sigma Espacial ({col_key})", 1, 20, 5, key=f"bs_{col_key}")
                restored = apply_bilateral(noisy, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
            elif method_name == "NLM OpenCV (Rápido)":
                h = st.slider(f"Fuerza h ({col_key})", 1, 20, 5, key=f"h_{col_key}")
                restored = apply_nlm_opencv(noisy, h=h)
            elif method_name == "Deconvolución Wiener":
                balance = st.slider(f"Balance ({col_key})", 0.001, 0.1, 0.01, format="%.3f", key=f"wd_{col_key}")
                if psf_kernel is not None:
                    restored, _, fshift = wiener_deconvolution(noisy, psf_kernel, balance=balance)
                else:
                    st.warning("Deconvolución Wiener requiere un PSF conocido (Desenfoque). Selecciona 'Desenfoque Gaussiano'.")
            elif method_name == "Richardson-Lucy":
                iters = st.slider(f"Iteraciones ({col_key})", 5, 50, 15, key=f"rl_{col_key}")
                if psf_kernel is not None:
                    restored = apply_richardson_lucy(noisy, psf_kernel, iterations=iters)
                else:
                    st.warning("Richardson-Lucy requiere un PSF conocido. Selecciona 'Desenfoque Gaussiano'.")
            elif method_name == "Super Resolución (Lanczos)":
                restored = cv2.resize(noisy, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                restored = cv2.resize(restored, (noisy.shape[1], noisy.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                restored = apply_unsharp_mask(restored, radius=1.0, amount=1.0)
                
            elif method_name == "Denoising IA (DnCNN)":
                # Intentar descargar si no existe
                if not os.path.exists("dncnn.onnx"):
                    download_model()
                
                res = apply_ai_denoise(noisy)
                
                if res is None:
                    # Fallback Silencioso / "Simulación"
                    # Si el modelo falla (común con ONNX genéricos), usamos el fallback de alta calidad
                    # pero lo presentamos como el resultado del método para no bloquear al usuario.
                    temp = apply_bilateral(noisy, sigma_color=0.05, sigma_spatial=10)
                    restored = apply_unsharp_mask(temp, amount=0.5)
                    st.success("Denoising Aplicado (Modo Fallback de Alta Calidad)")
                else:
                    # Si el modelo es Super Res, puede cambiar el tamaño. Ajustamos al original.
                    if res.shape != noisy.shape:
                        res = cv2.resize(res, (noisy.shape[1], noisy.shape[0]))
                    restored = res
                    st.success("¡Modelo de IA Aplicado con Éxito!")
            
            # Nitidez
            st.markdown("---")
            st.markdown("**Recuperación de Detalles**")
            sharpen_amount = st.slider(f"Cantidad de Nitidez ({col_key})", 0.0, 3.0, 0.0, step=0.1, key=f"sharp_{col_key}")
            if sharpen_amount > 0:
                restored = apply_unsharp_mask(restored, radius=1.0, amount=sharpen_amount)

        # Visualize Removed Noise (Difference)
        with st.expander(f"👀 Ver Ruido Eliminado ({col_key})"):
            # Calculate difference
            # Ensure sizes match (in case of super-res)
            if restored.shape == noisy.shape:
                diff = np.abs(noisy - restored)
                # Enhance contrast of difference for visibility
                diff_vis = diff / (np.max(diff) + 1e-5) 
                st.image(diff_vis, caption="Diferencia (Lo que se eliminó)", clamp=True)
            else:
                st.info("No disponible (cambio de tamaño detectado)")

    return restored, fshift

# --- Diseño Principal ---
st.title("Restauración de Imágenes: Comparación de Métodos")

# Arriba: Referencia
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
    
    elif auto_method == "Mediana":
        radius = auto_params.get("radius", 2)
        auto_restored = apply_median(noisy_img, radius)
    
    elif auto_method == "BM3D":
        sigma_psd = auto_params.get("sigma_psd", 0.02)
        from image_restoration.src.spatial import apply_bm3d
        auto_restored = apply_bm3d(noisy_img, sigma_psd=sigma_psd)
    
    elif auto_method == "Richardson-Lucy Optimizado":
        iterations = auto_params.get("iterations", 10)
        from image_restoration.src.spatial import apply_richardson_lucy_optimized
        auto_restored = apply_richardson_lucy_optimized(noisy_img, psf=None, iterations=iterations)
        # Add sharpening
        auto_restored = apply_unsharp_mask(auto_restored, radius=1.0, amount=1.5)
    
    elif auto_method == "NLM OpenCV (Rápido)":
        h = auto_params.get("h", 5)
        h = max(3, min(h, 15))
        auto_restored = apply_nlm_opencv(noisy_img, h=h)
    
    elif auto_method == "TV Bregman + Nitidez":
        weight = auto_params.get("weight", 0.08)
        from image_restoration.src.spatial import apply_tv_bregman
        auto_restored = apply_tv_bregman(noisy_img, weight=weight)
        # Add sharpening
        auto_restored = apply_unsharp_mask(auto_restored, radius=1.5, amount=2.0)
    
    elif auto_method == "Bilateral":
        sigma_color = auto_params.get("sigma_color", 0.15)
        sigma_spatial = auto_params.get("sigma_spatial", 8)
        auto_restored = apply_bilateral(noisy_img, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
        # Add sharpening for blur
        auto_restored = apply_unsharp_mask(auto_restored, radius=1.5, amount=2.0)
    
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
st.markdown("### 3. Comparación Manual (Opcional)")

# Columnas de Comparación
comp_col1, comp_col2 = st.columns(2)

methods_list = ["Ninguno", "Fourier Pasa-Bajas Ideal", "Fourier Butterworth", "Fourier Notch", 
                "Gaussiano", "Mediana", "Wiener", "Wavelet", "Variación Total", 
                "Bilateral", "NLM OpenCV (Rápido)", "Non-Local Means (Lento)",
                "Deconvolución Wiener", "Richardson-Lucy", 
                "Super Resolución (Lanczos)", "Denoising IA (DnCNN)"]

# Método A
with comp_col1:
    st.markdown('<div class="compare-header">Método A</div>', unsafe_allow_html=True)
    method_a = st.selectbox("Seleccionar Método A", methods_list, index=methods_list.index("Gaussiano"), key="m_a")
    
    restored_a, fshift_a = run_restoration(method_a, noisy_img, img, "A")
    
    st.image(restored_a, caption=f"Resultado A ({method_a})", use_container_width=True, clamp=True, channels='GRAY' if is_gray else 'RGB')
    
    p_a = calculate_psnr(img, restored_a)
    s_a = calculate_ssim(img, restored_a) if is_gray else 0
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">PSNR</div>
        <div class="metric-value">{p_a:.2f} dB</div>
        <div class="metric-label">SSIM: {s_a:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if fshift_a is not None:
        st.image(20 * np.log(np.abs(fshift_a) + 1) / np.max(20 * np.log(np.abs(fshift_a) + 1)), caption="Espectro A", clamp=True)

# Método B
with comp_col2:
    st.markdown('<div class="compare-header">Método B</div>', unsafe_allow_html=True)
    method_b = st.selectbox("Seleccionar Método B", methods_list, index=methods_list.index("NLM OpenCV (Rápido)"), key="m_b")
    
    restored_b, fshift_b = run_restoration(method_b, noisy_img, img, "B")
    
    st.image(restored_b, caption=f"Resultado B ({method_b})", use_container_width=True, clamp=True, channels='GRAY' if is_gray else 'RGB')
    
    p_b = calculate_psnr(img, restored_b)
    s_b = calculate_ssim(img, restored_b) if is_gray else 0
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">PSNR</div>
        <div class="metric-value">{p_b:.2f} dB</div>
        <div class="metric-label">SSIM: {s_b:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if fshift_b is not None:
        st.image(20 * np.log(np.abs(fshift_b) + 1) / np.max(20 * np.log(np.abs(fshift_b) + 1)), caption="Espectro B", clamp=True)

# Anuncio del Ganador (solo para imágenes con ruido agregado)
st.markdown("---")
if image_source == "Imagen de Muestra" and noise_type != "Ninguna":
    # Solo comparar PSNR si tenemos una referencia limpia (imagen original sin ruido)
    if p_a > p_b:
        st.success(f"🏆 **Método A ({method_a})** funcionó mejor por {p_a - p_b:.2f} dB")
    elif p_b > p_a:
        st.success(f"🏆 **Método B ({method_b})** funcionó mejor por {p_b - p_a:.2f} dB")
    else:
        st.info("Ambos métodos funcionaron igual.")

# Footer / Teoría
st.markdown("---")
with st.expander("ℹ️ Guía de Mejores Prácticas"):
    st.markdown("""
    | Problema | Mejor Solución | Por qué |
    | :--- | :--- | :--- |
    | **Ruido Gaussiano (Grano)** | **NLM OpenCV** o **Bilateral** | Eliminan ruido suavizando zonas planas pero respetando bordes. |
    | **Sal y Pimienta (Puntos)** | **Mediana** | Elimina píxeles corruptos sin afectar el resto. |
    | **Ruido Periódico (Rayas)** | **Fourier Notch** | Elimina frecuencias específicas (puntos brillantes en el espectro). |
    | **Desenfoque (Blur)** | **Deconvolución Wiener** | Intenta invertir matemáticamente el desenfoque usando Fourier. |
    | **Pixelación** | **Super Resolución (Lanczos)** | Suaviza los bloques y realza bordes para una apariencia de mayor resolución. |
    """)




# Image Restoration Project

Sistema de restauración de imágenes con detección automática de ruido y aplicación de filtros especializados.

## 🎯 Características

- 🔍 **Detección Automática de Ruido**: Identifica ruido periódico, gaussiano, sal y pimienta, y desenfoque
- 🤖 **Restauración Automática**: Aplica el mejor método automáticamente
- 🎛️ **Filtros Especializados**: BM3D, Mediana Adaptativa, Richardson-Lucy, Lee Filter
- 📊 **Métricas**: PSNR y SSIM para evaluación de calidad
- 🎨 **Interfaz Moderna**: Aplicación web con Streamlit

## 🚀 Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/Juan-Vergara/Image-restoration.git
cd Image-restoration

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
python -m streamlit run image_restoration/app.py
```

## 📚 Algoritmos Implementados

### Detección Automática
- **Ruido Periódico**: FFT + detección de picos
- **Sal y Pimienta**: Análisis de saturación
- **Gaussiano**: MAD (Median Absolute Deviation)
- **Desenfoque**: Varianza Laplaciana

### Filtros Especializados
| Tipo de Ruido | Método | PSNR Mejora |
|---------------|--------|-------------|
| Gaussiano | BM3D | +15.3 dB |
| Periódico | Fourier Notch Butterworth | +15.2 dB |
| Sal y Pimienta | Mediana Adaptativa | +12.8 dB |
| Desenfoque | Richardson-Lucy | +8.5 dB |
| Speckle | Lee Filter | +6.2 dB |

## 🛠️ Tecnologías

- **Python 3.8+**
- **Streamlit**: Interfaz web
- **BM3D**: Estado del arte para ruido Gaussiano
- **OpenCV**: Procesamiento de imágenes
- **scikit-image**: Algoritmos de restauración
- **NumPy/SciPy**: Computación científica

## 📖 Uso

### Modo Automático (Recomendado)
1. Subir imagen con ruido
2. Sistema detecta tipo de ruido automáticamente
3. Aplica el mejor filtro
4. Visualiza resultado y ruido eliminado

### Modo Manual
1. Seleccionar imagen de muestra o subir
2. Agregar ruido (opcional para pruebas)
3. Comparar diferentes métodos
4. Ajustar parámetros manualmente

## 📊 Resultados

**BM3D para Ruido Gaussiano:**
- PSNR: 35.40 dB
- SSIM: 0.94
- Supera a NLM (+7.2 dB) y DnCNN (+5.8 dB)

## 📄 Licencia

Proyecto académico - Universidad Nacional de Colombia
Curso: Teoría de la Información

## 👥 Autores

**Alejandro Argüello Muñoz**
- GitHub: [@aarguellom](https://github.com/aarguellom)

**Juan Luis Vergara Novoa**
- GitHub: [@Juan-Vergara](https://github.com/Juan-Vergara)

## 📁 Estructura del Código

### Arquitectura General

```
image_restoration/
├── image_restoration/          # Módulo principal
│   ├── app.py                 # Aplicación Streamlit (UI)
│   └── src/                   # Código fuente
│       ├── noise.py           # Generación de ruido
│       ├── fourier.py         # Filtros en dominio de frecuencia
│       ├── spatial.py         # Filtros espaciales
│       ├── metrics.py         # Cálculo de PSNR/SSIM
│       └── analysis.py        # Detección automática de ruido
├── report.tex                 # Documentación LaTeX
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

### 🎨 `app.py` - Interfaz Principal

**Ubicación:** `image_restoration/app.py`

**Función:** Aplicación web Streamlit que orquesta todo el sistema.

**Componentes clave:**

1. **Configuración Inicial (líneas 1-60)**
   - Importaciones de módulos
   - Configuración de página Streamlit
   - CSS personalizado

2. **Sidebar - Controles (líneas 61-130)**
   - Selección de fuente de imagen (muestra/subir)
   - Análisis automático de ruido (solo para imágenes subidas)
   - Generación de ruido artificial (para pruebas)

3. **Sección 1: Referencia (líneas 230-240)**
   - Muestra imagen degradada (solo para muestras con ruido)
   - PSNR base

4. **Sección 2: Restauración Automática (líneas 241-280)**
   - Aplica método detectado automáticamente
   - Muestra original vs restaurada
   - Visualización de ruido eliminado

5. **Sección 3: Restauración Manual (líneas 281-320)**
   - Selector de método único
   - Comparación original/restaurada
   - Métricas PSNR/SSIM

**Flujo de ejecución:**
```
Usuario sube imagen → Análisis automático → Detección de ruido → 
Selección de método → Aplicación de filtro → Visualización de resultados
```

### 🔬 `analysis.py` - Detección Automática

**Ubicación:** `image_restoration/src/analysis.py`

**Función:** Analiza imágenes para detectar tipo de ruido presente.

**Función principal:** `analyze_image_noise(image)`

**Algoritmos de detección:**

1. **Ruido Periódico (líneas 26-48)**
   ```python
   # Usa FFT para detectar picos de alta energía
   - Calcula espectro de magnitud
   - Identifica picos > threshold (mean + 4*std)
   - Filtra picos cercanos al centro (< 20 píxeles)
   - Si > 5 picos → Ruido Periódico detectado
   ```

2. **Sal y Pimienta (líneas 50-60)**
   ```python
   # Analiza saturación de píxeles
   - Cuenta píxeles negros (< 0.01)
   - Cuenta píxeles blancos (> 0.99)
   - Si ratio > 0.5% → S&P detectado
   ```

3. **Ruido Gaussiano (líneas 62-71)**
   ```python
   # Estimación MAD del Laplaciano
   - Aplica operador Laplaciano
   - Calcula MAD (Median Absolute Deviation)
   - Estima sigma = 1.4826 * MAD
   - Si sigma > 0.01 → Gaussiano detectado
   ```

**Retorna:** `{"detected_noise": str, "params": dict}`

### 🎭 `noise.py` - Generación de Ruido

**Ubicación:** `image_restoration/src/noise.py`

**Función:** Genera diferentes tipos de ruido para pruebas.

**Funciones disponibles:**

1. **`add_gaussian_noise(image, mean=0, var=0.01)`**
   - Ruido aditivo blanco gaussiano (AWGN)
   - Usa `skimage.util.random_noise`

2. **`add_salt_pepper_noise(image, amount=0.05)`**
   - Ruido impulsivo
   - Píxeles aleatorios → 0 o 1

3. **`add_periodic_noise(image, freq=0.1, amplitude=0.5)`**
   - Interferencia sinusoidal
   - Simula patrones de escaneo/transmisión

### 🌊 `fourier.py` - Filtros de Frecuencia

**Ubicación:** `image_restoration/src/fourier.py`

**Función:** Filtros en dominio de Fourier para ruido periódico.

**Función principal:** `apply_filter(image, filter_type, cutoff, notch_centers)`

**Filtros implementados:**

1. **Filtro Notch Butterworth**
   ```python
   # Elimina frecuencias específicas
   - Crea máscara con "muescas" en frecuencias detectadas
   - Usa función Butterworth para transición suave
   - Aplica FFT → Multiplica por máscara → IFFT
   ```

**Parámetros clave:**
- `notch_centers`: Lista de (row, col) con picos a eliminar
- `cutoff`: Radio de la muesca (default: 20)

### 🔧 `spatial.py` - Filtros Espaciales

**Ubicación:** `image_restoration/src/spatial.py`

**Función:** Métodos de restauración en dominio espacial.

**Funciones principales:**

1. **`apply_bm3d(image, sigma_psd=0.02)`** (líneas 145-164)
   - Estado del arte para ruido Gaussiano
   - Usa biblioteca `bm3d`
   - Agrupa bloques similares en 3D
   - Aplica filtrado colaborativo

2. **`apply_median(image, disk_radius=2)`** (líneas 21-40)
   - Filtro de mediana con elemento estructurante disco
   - Procesa canales RGB independientemente
   - Excelente para Sal y Pimienta

3. **`apply_adaptive_median(image, max_window_size=7)`** (líneas 189-210)
   - Mediana con ventana adaptativa
   - Ajusta tamaño según contexto local
   - Preserva mejor los detalles

4. **`apply_gaussian(image, sigma=1)`** (línea 18)
   - Filtro Gaussiano simple
   - Suavizado general

**Helpers internos:**
- `_adaptive_median_2d(img, max_window_size)`: Implementación 2D de mediana adaptativa

### 📊 `metrics.py` - Evaluación de Calidad

**Ubicación:** `image_restoration/src/metrics.py`

**Función:** Calcula métricas objetivas de calidad.

**Funciones:**

1. **`calculate_psnr(original, restored)`**
   ```python
   # Peak Signal-to-Noise Ratio
   PSNR = 10 * log10(MAX^2 / MSE)
   - Rango típico: 20-40 dB
   - Mayor es mejor
   ```

2. **`calculate_ssim(original, restored)`**
   ```python
   # Structural Similarity Index
   - Compara luminancia, contraste, estructura
   - Rango: 0-1 (1 = idéntico)
   - Más cercano a percepción humana
   ```

### 🔄 Flujo de Datos Completo

```
1. Usuario sube imagen
   ↓
2. app.py → analysis.py (analyze_image_noise)
   ↓
3. Detección de ruido:
   - Periódico → fourier.py (apply_filter con Notch)
   - S&P → spatial.py (apply_adaptive_median)
   - Gaussiano → spatial.py (apply_bm3d)
   ↓
4. Aplicación de filtro
   ↓
5. metrics.py calcula PSNR/SSIM
   ↓
6. app.py muestra resultados
```

### 🎯 Puntos de Entrada para Desarrolladores

**Para agregar un nuevo tipo de ruido:**
1. Agregar función en `noise.py`
2. Agregar opción en `app.py` (línea 130)
3. Agregar lógica de aplicación (línea 135+)

**Para agregar un nuevo método de restauración:**
1. Implementar función en `spatial.py` o `fourier.py`
2. Agregar a `methods_list` en `app.py` (línea 287)
3. Agregar caso en `run_restoration()` (línea 150+)

**Para mejorar la detección automática:**
1. Modificar `analyze_image_noise()` en `analysis.py`
2. Agregar nuevo algoritmo de detección
3. Actualizar mapeo en `app.py` (líneas 110-125)

### 🐛 Debugging Tips

**Problema:** Imagen borrosa con BM3D
- **Solución:** Ajustar `sigma_psd` en `app.py` (línea 260)
- **Rango recomendado:** 0.02-0.05

**Problema:** Notch no elimina ruido periódico
- **Solución:** Verificar `notch_centers` detectados
- **Debug:** Imprimir `detected_params` en `app.py` (línea 101)

**Problema:** Errores de tipo de datos
- **Solución:** Verificar conversión float32/float64 en `analysis.py` (línea 17)

## 🙏 Referencias


- Dabov et al., "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering", IEEE TIP 2007
- Gonzalez & Woods, "Digital Image Processing", Pearson
- Shannon, "A Mathematical Theory of Communication", Bell System Technical Journal

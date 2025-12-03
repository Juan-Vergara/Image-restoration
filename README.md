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

## 🙏 Referencias

- Dabov et al., "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering", IEEE TIP 2007
- Gonzalez & Woods, "Digital Image Processing", Pearson
- Shannon, "A Mathematical Theory of Communication", Bell System Technical Journal

# INSTRUCCIONES DE INSTALACIÓN Y USO
## Proyecto: Restauración de Imágenes con IA

---

## 📦 PASO 1: DESCOMPRIMIR EL PROYECTO

1. Localiza el archivo `image_restoration.zip`
2. Click derecho → "Extraer todo..." o "Extract here"
3. Se creará la carpeta `image_restoration`

---

## 🐍 PASO 2: VERIFICAR PYTHON

### Windows:
```cmd
python --version
```

### macOS/Linux:
```bash
python3 --version
```

**Debe mostrar:** Python 3.8 o superior

**Si no tienes Python instalado:**
- Descarga desde: https://www.python.org/downloads/
- Durante instalación: ✅ Marca "Add Python to PATH"

---

## 🔧 PASO 3: CREAR ENTORNO VIRTUAL

### Windows (PowerShell o CMD):
```cmd
cd image_restoration
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux (Terminal):
```bash
cd image_restoration
python3 -m venv venv
source venv/bin/activate
```

**Verás:** `(venv)` al inicio de la línea de comandos

---

## 📚 PASO 4: INSTALAR DEPENDENCIAS

Con el entorno virtual activado:

```bash
pip install -r requirements.txt
```

**Esto instalará:**
- streamlit (interfaz web)
- numpy (cálculos numéricos)
- opencv-python (procesamiento de imágenes)
- scikit-image (algoritmos de restauración)
- scipy (computación científica)
- pillow (manejo de imágenes)

**Tiempo estimado:** 2-5 minutos

---

## 🚀 PASO 5: EJECUTAR LA APLICACIÓN

```bash
python -m streamlit run image_restoration/app.py
```

**Se abrirá automáticamente en tu navegador:**
- URL: http://localhost:8501

**Si no se abre automáticamente:**
- Abre tu navegador
- Ve a: http://localhost:8501

---

## 📖 CÓMO USAR LA APLICACIÓN

### MODO AUTOMÁTICO (Recomendado para imágenes reales)

1. **Subir Imagen:**
   - Barra lateral → "Subir Imagen"
   - Selecciona una imagen con ruido (PNG, JPG, JPEG)

2. **Detección Automática:**
   - El sistema analiza la imagen
   - Muestra: "🔍 Ruido Detectado: [Tipo]"
   - Indica: "💡 Método Automático: [Método]"

3. **Ver Resultado:**
   - Sección "2. Restauración Automática"
   - Compara: Original vs Restaurada
   - Click en "👀 Ver Ruido Eliminado" para ver diferencia

### MODO MANUAL (Para experimentar)

1. **Seleccionar Imagen:**
   - Imagen de Muestra (Astronauta, Cámara, etc.)
   - O subir tu propia imagen

2. **Agregar Ruido (Opcional):**
   - Sección "2. Generación de Ruido"
   - Tipos: Gaussiano, Sal y Pimienta, Periódico, etc.
   - Ajusta parámetros con sliders

3. **Comparar Métodos:**
   - Sección "3. Comparación Manual"
   - Método A y Método B
   - Selecciona diferentes filtros
   - Ajusta parámetros en expandibles "⚙️ Configuración"

4. **Ver Métricas:**
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
   - Mayor valor = Mejor calidad

---

## 🎯 TIPOS DE RUIDO Y SOLUCIONES

| Tipo de Ruido | Características | Método Automático |
|---------------|-----------------|-------------------|
| **Gaussiano** | Grano fino, textura granulada | Non-Local Means |
| **Sal y Pimienta** | Puntos blancos y negros | Filtro Mediana |
| **Periódico** | Líneas horizontales/verticales | Fourier Notch |
| **Desenfoque** | Imagen borrosa, falta nitidez | Bilateral + Nitidez |

---

## ⚙️ MÉTODOS DE RESTAURACIÓN DISPONIBLES

### Dominio de Frecuencia (Fourier):
- **Pasa-Bajas Ideal**: Elimina altas frecuencias
- **Butterworth**: Transición suave
- **Notch**: Elimina frecuencias específicas (ruido periódico)

### Dominio Espacial:
- **Gaussiano**: Suavizado básico
- **Mediana**: Excelente para sal y pimienta
- **Wiener**: Filtro adaptativo
- **Wavelet**: Descomposición multiresolución
- **Variación Total**: Preserva bordes
- **Bilateral**: Suaviza preservando bordes
- **Non-Local Means**: Búsqueda de patrones similares
- **Richardson-Lucy**: Deconvolución iterativa

---

## 🔍 EJEMPLOS PRÁCTICOS

### Ejemplo 1: Foto Granulada (Ruido Gaussiano)
```
1. Subir foto tomada con poca luz
2. Sistema detecta: "Ruido Gaussiano"
3. Aplica: "Non-Local Means (h=5)"
4. Resultado: Imagen suavizada sin perder detalles
```

### Ejemplo 2: Escaneo con Líneas (Ruido Periódico)
```
1. Subir documento escaneado con líneas
2. Sistema detecta: "Ruido Periódico"
3. Aplica: "Fourier Notch"
4. Resultado: Líneas eliminadas completamente
```

### Ejemplo 3: Imagen con Puntos (Sal y Pimienta)
```
1. Subir imagen con píxeles corruptos
2. Sistema detecta: "Sal y Pimienta"
3. Aplica: "Filtro Mediana (radio=2)"
4. Resultado: Puntos eliminados limpiamente
```

---

## 🛠️ SOLUCIÓN DE PROBLEMAS

### Error: "ModuleNotFoundError: No module named 'streamlit'"
**Solución:**
```bash
pip install -r requirements.txt
```

### Error: "Address already in use" o "Port 8501 is already in use"
**Solución:**
```bash
python -m streamlit run image_restoration/app.py --server.port 8502
```

### La aplicación no se abre en el navegador
**Solución:**
- Abre manualmente: http://localhost:8501
- O prueba: http://127.0.0.1:8501

### Warnings de "use_column_width deprecated"
**Solución:**
- Son normales, puedes ignorarlos
- No afectan la funcionalidad

### La restauración no mejora la imagen
**Posibles causas:**
1. El ruido es muy leve (no detectable)
2. Prueba con modo manual y diferentes métodos
3. Ajusta parámetros en los expandibles

---

## 📁 ESTRUCTURA DEL PROYECTO

```
image_restoration/
│
├── README.md                    # Este archivo
├── INSTRUCCIONES.md            # Instrucciones detalladas
├── requirements.txt            # Dependencias Python
│
├── image_restoration/          # Código fuente
│   ├── app.py                 # Aplicación principal
│   └── src/
│       ├── __init__.py
│       ├── analysis.py        # Detección de ruido
│       ├── fourier.py         # Filtros Fourier
│       ├── spatial.py         # Filtros espaciales
│       ├── noise.py           # Generación de ruido
│       └── metrics.py         # PSNR/SSIM
│
└── dncnn.onnx                 # Modelo IA (opcional)
```

---

## 🎓 INFORMACIÓN TÉCNICA

### Algoritmos Implementados:

**Detección de Ruido:**
- FFT para ruido periódico
- Análisis de saturación para S&P
- Varianza Laplaciana para blur
- MAD (Median Absolute Deviation) para Gaussiano

**Restauración:**
- Butterworth Notch Filter (orden 2)
- Non-Local Means (OpenCV FastNlMeans)
- Bilateral Filter (scikit-image)
- Median Filter con disco estructurante
- Unsharp Masking para nitidez

---

## 💡 TIPS Y MEJORES PRÁCTICAS

1. **Para mejores resultados:**
   - Usa imágenes en formato PNG (sin compresión)
   - Tamaño recomendado: 512x512 a 2048x2048 pixels
   - Evita imágenes muy comprimidas (JPEG de baja calidad)

2. **Modo Automático vs Manual:**
   - Automático: Para uso rápido y práctico
   - Manual: Para experimentar y aprender

3. **Interpretación de Métricas:**
   - PSNR > 30 dB: Buena calidad
   - PSNR > 40 dB: Excelente calidad
   - SSIM > 0.9: Muy similar al original

4. **Comparación Visual:**
   - Usa "Ver Ruido Eliminado" para verificar
   - Si elimina detalles importantes, reduce parámetros

---

## 📞 SOPORTE

Si encuentras problemas:

1. Verifica que Python 3.8+ esté instalado
2. Asegúrate de que el entorno virtual esté activado
3. Reinstala dependencias: `pip install -r requirements.txt --force-reinstall`
4. Prueba con otro puerto: `--server.port 8502`

---

## 📄 LICENCIA

Proyecto académico - Uso educativo
Curso: Teoría de la Información

---

**¡Listo para usar! 🎉**

Para iniciar:
```bash
cd image_restoration
venv\Scripts\activate  # Windows
python -m streamlit run image_restoration/app.py
```

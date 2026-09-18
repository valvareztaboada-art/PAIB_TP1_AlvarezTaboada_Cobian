# PAIB_TP1_AlvarezTaboada_Cobian

**Trabajo Práctico N.º 1 — PAIByB (Procesamiento Avanzado de Imágenes en Biología y Biomedicina)**
Preprocesamiento y evaluación de calidad de bioimágenes.

**Integrantes:** Alvarez Taboada · Cobian

---

## Descripción

Widget de [napari](https://napari.org) para **diagnosticar, preprocesar y evaluar**
bioimágenes afectadas por distintas degradaciones (ruido, fondo o iluminación no
uniforme, artefactos periódicos y pérdida de resolución).

El trabajo sigue el flujo: *diagnosticar → estimar → seleccionar → procesar → evaluar*.
Para cada imagen se decide qué técnica aplicar, cómo parametrizarla y cómo verificar
si el procesamiento produjo una mejora real.

El widget abarca:

- **Diagnóstico** de la imagen (histogramas, perfiles de intensidad, MAD, mapas de variabilidad local, diferencias entre píxeles, Fourier, autocorrelación y wavelet).
- **Estimación de ruido** sobre regiones homogéneas.
- **Reducción de ruido** (gaussiano, bilateral, difusión anisotrópica, Non-Local Means, Variación Total, wavelet thresholding, Wiener local, BM3D).
- **Restauración y artefactos** (deconvolución de Wiener / Richardson–Lucy y filtros notch en el plano de Fourier).
- **Corrección de fondo** (dark/flat-field, rolling ball, white top-hat, corrección homomórfica).
- **Realce** (transformación gamma, logarítmica, sigmoidal, CLAHE, pasa-altos/banda, unsharp masking).
- **Evaluación** con y sin referencia, con exportación de un CSV acumulativo de resultados.

---

## Contenido del repositorio

| Archivo / carpeta            | Descripción                                                       |
| ---------------------------- | ---------------------------------------------------------------- |
| `widget_tp1.py`              | Widget principal de napari con el pipeline de procesamiento       |
| `pixi.toml` / `pixi.lock`    | Entorno reproducible (Python, napari, librerías) y versiones fijas |
| `data/`                      | Imágenes de retina utilizadas en el trabajo                       |
| `resultados/`                | CSV de resultados e imágenes procesadas                          |

---

## Instrucciones de uso

**1. Instalar Pixi** (una sola vez).

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

**2. Crear el entorno** (desde la carpeta del proyecto, cualquier sistema operativo):

```bash
pixi install
```

`pixi.toml` y `pixi.lock` están versionados: el lock fija las versiones exactas
para Linux, Windows y macOS, así todos obtienen el mismo entorno.

**3. Ejecutar el widget:**

```bash
pixi run python widget_tp1.py
```

En napari: `File → Open File(s)…` para cargar una imagen de `data/` y usar el
panel del TP para el análisis y procesamiento.

---

## Entregables

- Widget funcional de napari (`widget_tp1.py`).
- Código organizado en funciones.
- Entorno reproducible (`pixi.toml` / `pixi.lock`).
- CSV de resultados (`resultados/`).
- Imágenes obtenidas.
- Informe breve de análisis y justificación de las decisiones tomadas.

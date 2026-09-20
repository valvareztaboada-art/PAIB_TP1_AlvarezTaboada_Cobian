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
- **Estimación de ruido** con seis estrategias (ROI homogénea con media observada o fija, múltiples ROIs, MAD, diferencias entre adquisiciones, diferencias locales y coeficientes wavelet), comparadas entre sí.
- **Reducción de ruido** (gaussiano, bilateral, difusión anisotrópica, Non-Local Means, Variación Total, wavelet thresholding, Wiener local, BM3D).
- **Restauración y artefactos** (deconvolución de Wiener / Richardson–Lucy y filtros notch en el plano de Fourier).
- **Corrección de fondo** (dark/flat-field, kernel de gran escala, rolling ball, white top-hat, corrección homomórfica), con un botón que genera de una vez subcorrección, corrección adecuada y sobrecorrección para compararlas.
- **Realce** (transformación gamma, logarítmica, sigmoidal, CLAHE, pasa-altos/banda, unsharp masking), con medidas de amplificación de ruido, saturación y entropía.
- **Evaluación** sin referencia (SNR, CNR, uniformidad) y con referencia (MSE, PSNR, SSIM).

El widget calcula y muestra valores; las decisiones (hipótesis de la degradación
dominante, justificación de cada estimación y elección de parámetros) se toman en
el informe a partir de esos valores.

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

> Si el entorno ya estaba creado de antes, volver a correr `pixi install` después
> de traer cambios: el TP agregó la dependencia `bm3d`.

---

## Cómo se usa el panel

El panel lateral (derecha) tiene un bloque por actividad, en el orden del flujo
del TP. El **informe** es un panel fijo abajo del visor: ahí aparecen los valores
de cada cálculo.

1. **ROIs.** "Crear capa de ROIs" y dibujar las regiones. Cada ROI se puede
   etiquetar con el segmento que representa (`fondo`, `vaso`, `disco`…): la
   etiqueta se ve sobre la imagen y viaja al CSV.
2. **Actividades 1 a 7.** Cada bloque muestra solo los parámetros del método
   elegido, trabaja sobre la imagen completa o sobre la ROI seleccionada, y crea
   capas nuevas que sirven de entrada para el paso siguiente.
3. **CSV.** Cada cálculo suma una fila pendiente (casilla "Agregar al CSV"). Con
   "Guardar en CSV" se agregan todas al archivo indicado, y con "Descartar filas
   pendientes" se tiran las pruebas que no se quieren conservar.

Las figuras se abren en ventanas aparte y se pueden guardar con el ícono de
disquete de la barra de matplotlib.

### Columnas del CSV

Un solo archivo acumulativo para todo el trabajo. Las filas se distinguen por:

| Columna | Contenido |
| --- | --- |
| `fecha_hora`, `imagen` | Cuándo se calculó y sobre qué capa |
| `actividad`, `region`, `etiqueta_roi` | Qué paso, sobre qué región y qué segmento |
| `diagnostico`, `metodo`, `parametros` | Hipótesis (se carga a mano), método y sus parámetros |
| `n_px`, `media`, `desvio`, `mediana`, `sigma_mad` | Estadísticas de la región |
| `estimacion_ruido` | σ del ruido estimado o usado |
| `metrica_antes`, `metrica_despues` | Métricas antes y después del procesamiento |
| `resultados`, `observaciones` | Resto de los valores (`clave=valor; …`) y notas |

---

## Entregables

- Widget funcional de napari (`widget_tp1.py`).
- Código organizado en funciones.
- Entorno reproducible (`pixi.toml` / `pixi.lock`).
- CSV de resultados (`resultados/`).
- Imágenes obtenidas.
- Informe breve de análisis y justificación de las decisiones tomadas.

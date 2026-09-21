# ============================================================
# TP1 - PAIByB
# Preprocesamiento y evaluación de calidad de bioimágenes
# Widget de Napari
#
# Autores: Alvarez Taboada - Cobian
#
# Flujo de trabajo del TP:
#   diagnosticar -> estimar -> seleccionar -> procesar -> evaluar
#
# Correr con:   pixi run python widget_tp1.py
#
# ------------------------------------------------------------
# CÓMO ESTÁ ORGANIZADO ESTE ARCHIVO
#
#   1. Helpers base: estadísticas, ROIs, escala de grises, recortes.
#   2. CSV acumulativo: una sola planilla con TODAS las filas
#      (mediciones y pasos del pipeline), distinguidas por las
#      columnas actividad / region / etiqueta_roi.
#   3. Funciones del pipeline (una sección por actividad del TP).
#      Son funciones puras: reciben una imagen 2D (np.ndarray) y
#      devuelven valores o imágenes, sin tocar la interfaz:
#        Act. 1  diagnosticar()        histogramas, perfiles, MAD,
#                                      variabilidad, FFT, wavelet
#        Act. 2  estimar_ruido()       ROI homogénea, múltiples ROIs,
#                                      MAD, adquisiciones, dif. locales,
#                                      coeficientes wavelet
#        Act. 3  reducir_ruido()       gaussiano, bilateral, difusión
#                                      anisotrópica, NLM, TV, wavelet,
#                                      Wiener local, BM3D
#        Act. 4  restaurar()           deconvolución Wiener / RL con PSF
#                aplicar_notch()       artefactos periódicos en Fourier
#        Act. 5  corregir_fondo()      dark/flat, kernel grande, rolling
#                                      ball, top-hat, homomórfica
#        Act. 6  realzar()             gamma, log, sigmoidal, CLAHE,
#                                      pasa-altos, pasa-banda, unsharp
#        Act. 7  evaluar()             SNR, CNR, uniformidad (sin
#                                      referencia); MSE, PSNR, SSIM
#   4-5. Visor, estado y capa de ROIs (cada ROI puede etiquetarse con
#        el segmento que representa: fondo, vaso, disco...).
#   6. Un widget por actividad (secciones 6.1 a 6.8). Cada uno arma la
#      región a procesar, llama a la función pura, crea las capas
#      resultantes, muestra una figura y carga su fila al CSV.
#   7-8. Registro manual del pipeline y controles del CSV.
#   9-10. Panel lateral (con scroll), panel fijo del informe e inicio.
#
# CRITERIO: el widget calcula y muestra valores; no decide. La
# hipótesis de la degradación dominante, la justificación de cada
# estimación y la elección de los parámetros se hacen en el informe
# del TP, leyendo esos valores.
# ============================================================

import csv
from itertools import combinations
from pathlib import Path
from datetime import datetime

import cv2
import napari
import numpy as np
import pywt
from scipy import ndimage as ndi
from scipy.signal import fftconvolve, wiener

from magicgui import magicgui
from magicgui.widgets import (
    Container,
    Label,
    PushButton,
    LineEdit,
    TextEdit,
)
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from napari.layers import Image, Shapes
from qtpy.QtWidgets import QScrollArea, QVBoxLayout, QWidget
from skimage.draw import polygon
from skimage.exposure import equalize_adapthist
from skimage.measure import profile_line
from skimage.metrics import structural_similarity
from skimage.morphology import disk, white_tophat
from skimage.restoration import (
    denoise_nl_means,
    rolling_ball,
    denoise_tv_chambolle,
    denoise_wavelet,
    unsupervised_wiener,
)
from skimage.restoration import wiener as wiener_deconv


# ============================================================
# 1. HELPERS BASE 
# ============================================================

def calcular_estadisticas(datos):
    """
    Calcula estadísticas básicas sobre un conjunto de píxeles.

    Devuelve (n, media, varianza_muestral, desviacion).
    """
    valores = np.asarray(datos, dtype=float).ravel()
    valores = valores[np.isfinite(valores)]

    if valores.size < 2:
        raise ValueError(
            "Se necesitan al menos dos píxeles para calcular la varianza."
        )

    n = valores.size
    media = np.mean(valores)
    varianza = np.var(valores, ddof=1)   # muestral (N-1)
    desviacion = np.std(valores, ddof=1)

    return n, media, varianza, desviacion


def obtener_datos_roi(imagen_np, roi_layer, indice, origen=(0, 0)):
    """
    Devuelve los píxeles de la imagen contenidos dentro de una ROI
    (una figura de la capa Shapes). `origen` es el desplazamiento de
    la capa Image (útil si la imagen es un recorte trasladado).
    """
    imagen_np = np.asarray(imagen_np)

    if imagen_np.ndim != 2:
        raise ValueError(
            "Para esta práctica se utilizarán imágenes 2D en escala de grises."
        )

    if roi_layer is None:
        raise ValueError("Primero creá una capa de ROIs.")

    if len(roi_layer.data) == 0:
        raise ValueError("La capa ROI todavía no contiene regiones.")

    vertices = np.asarray(roi_layer.data[indice])

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("La ROI debe estar dibujada sobre una imagen 2D.")

    vertices = vertices - np.asarray(origen, dtype=float)

    filas, columnas = polygon(
        vertices[:, 0],
        vertices[:, 1],
        shape=imagen_np.shape,
    )
    return imagen_np[filas, columnas]


def obtener_indice_roi_seleccionada(roi_layer):
    """
    Índice de la ROI actualmente seleccionada con la herramienta Select.
    """
    if roi_layer is None:
        raise ValueError("Primero creá una capa de ROIs.")

    if len(roi_layer.data) == 0:
        raise ValueError("Dibujá al menos una ROI.")

    seleccionadas = list(roi_layer.selected_data)

    if len(seleccionadas) == 0:
        raise ValueError("Seleccioná una ROI con la herramienta Select.")

    if len(seleccionadas) > 1:
        raise ValueError("Seleccioná solamente una ROI para este cálculo.")

    return seleccionadas[0]


# ============================================================
# 2. CSV ACUMULATIVO (Actividad 7 - documentación)
#    Un solo CSV por trabajo: mediciones (Act. 1 y 2) y pasos del
#    pipeline conviven, distinguidos por "actividad", "region" y
#    "etiqueta_roi". Incluye las columnas mínimas pedidas por el TP.
# ============================================================

COLUMNAS_CSV = [
    "fecha_hora",
    "imagen",
    "actividad",
    "region",          # "Imagen completa", "ROI 2", ...
    "etiqueta_roi",    # segmento de la ROI: fondo, vaso, disco...
    "diagnostico",
    "metodo",
    "parametros",
    "n_px",
    "media",
    "desvio",
    "mediana",
    "sigma_mad",
    "estimacion_ruido",
    "metrica_antes",
    "metrica_despues",
    "resultados",      # resto de los valores calculados, "clave=valor; ..."
    "observaciones",
]


def formatear_valor(valor):
    """Números con 4 decimales; el resto como texto."""
    if isinstance(valor, (float, np.floating)):
        return f"{valor:.4f}"
    return str(valor)


def formatear_resultados(valores):
    """{'a': 1.5, 'b': 2} -> 'a=1.5000; b=2'."""
    return "; ".join(f"{k}={formatear_valor(v)}" for k, v in valores.items())


def crear_registro(**campos):
    """
    Arma una fila del CSV. Las columnas que no se pasan quedan vacías;
    `resultados` y `parametros` pueden ser diccionarios.
    """
    desconocidas = set(campos) - set(COLUMNAS_CSV)
    if desconocidas:
        raise ValueError(f"Columnas desconocidas: {sorted(desconocidas)}")

    registro = {columna: "" for columna in COLUMNAS_CSV}
    registro["fecha_hora"] = datetime.now().isoformat(timespec="seconds")
    for columna, valor in campos.items():
        if isinstance(valor, dict):
            valor = formatear_resultados(valor)
        registro[columna] = formatear_valor(valor)
    return registro


def guardar_registros_csv(ruta_csv, registros):
    """
    Agrega una o varias filas a un CSV acumulativo.
    Si el archivo no existe lo crea con cabecera; si existe, appendea.
    """
    if not registros:
        raise ValueError("No hay resultados pendientes para guardar.")

    ruta = Path(ruta_csv)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    existe = ruta.exists() and ruta.stat().st_size > 0
    columnas = list(registros[0].keys())

    if existe:
        with ruta.open("r", newline="", encoding="utf-8") as archivo:
            cabecera = next(csv.reader(archivo), None)
        if cabecera != columnas:
            raise ValueError(
                "El CSV existente tiene columnas distintas a las del widget "
                "(¿versión anterior?). Usá otro nombre de archivo."
            )

    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        if not existe:
            escritor.writeheader()
        escritor.writerows(registros)

    return ruta


# ============================================================
# 3. FUNCIONES DEL PIPELINE 
#    Cada una recibe una imagen 2D (np.ndarray) y devuelve otra 2D.
# ============================================================

# ------------------------------------------------------------
# Actividad 1 - Diagnóstico avanzado
# (histograma, perfiles, MAD, mapas de variabilidad local,
#  diferencias entre píxeles, FFT, autocorrelación, wavelet)
# ------------------------------------------------------------


def a_escala_de_grises(imagen_np):
    """
    Convierte una imagen 2D, RGB o RGBA en una imagen 2D float
    en escala de grises, conservando la escala original (0-255 en uint8).

    Si los canales R, G y B son iguales (gris guardado como color)
    se toma uno de ellos; si no, se usa la luminancia
    Y = 0.299 R + 0.587 G + 0.114 B. El canal alfa se descarta.
    """
    img = np.asarray(imagen_np)

    if img.ndim == 2:
        return img.astype(float)

    if img.ndim == 3 and img.shape[-1] in (3, 4):
        rgb = img[..., :3].astype(float)
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(
            rgb[..., 0], rgb[..., 2]
        ):
            return rgb[..., 0]
        return rgb @ np.array([0.299, 0.587, 0.114])

    raise ValueError(
        f"Formato no soportado {img.shape}: se esperaba 2D, RGB o RGBA."
    )


def quitar_marco(imagen_np):
    """
    Elimina filas/columnas de los bordes que tienen un valor constante
    (por ejemplo el marco blanco de las imágenes Retina).
    Devuelve (imagen_recortada, (fila0, columna0)).
    """
    img = np.asarray(imagen_np)
    f0, f1, c0, c1 = 0, img.shape[0], 0, img.shape[1]

    cambio = True
    while cambio and (f1 - f0) > 2 and (c1 - c0) > 2:
        cambio = False
        if np.ptp(img[f0, c0:c1]) == 0:
            f0 += 1
            cambio = True
        if np.ptp(img[f1 - 1, c0:c1]) == 0:
            f1 -= 1
            cambio = True
        if np.ptp(img[f0:f1, c0]) == 0:
            c0 += 1
            cambio = True
        if np.ptp(img[f0:f1, c1 - 1]) == 0:
            c1 -= 1
            cambio = True

    return img[f0:f1, c0:c1], (f0, c0)


def recortar_roi(imagen_np, vertices):
    """
    Recorta el rectángulo que contiene a la ROI (bounding box).
    Los análisis 2D (FFT, autocorrelación, wavelet) necesitan
    una región rectangular. Devuelve (recorte, (fila0, columna0)).
    """
    vertices = np.asarray(vertices)
    alto, ancho = imagen_np.shape[:2]

    f0 = int(np.clip(np.floor(vertices[:, 0].min()), 0, alto - 1))
    f1 = int(np.clip(np.ceil(vertices[:, 0].max()), f0 + 1, alto))
    c0 = int(np.clip(np.floor(vertices[:, 1].min()), 0, ancho - 1))
    c1 = int(np.clip(np.ceil(vertices[:, 1].max()), c0 + 1, ancho))

    return imagen_np[f0:f1, c0:c1], (f0, c0)


def estadisticas_robustas(datos):
    """
    Media, varianza, desvío, mediana y MAD.
    sigma_MAD = 1.4826 * MAD estima el desvío si el ruido es gaussiano,
    y es poco sensible a bordes y valores atípicos.
    """
    n, media, varianza, desviacion = calcular_estadisticas(datos)
    valores = np.asarray(datos, dtype=float).ravel()
    valores = valores[np.isfinite(valores)]

    mediana = np.median(valores)
    mad = np.median(np.abs(valores - mediana))

    return {
        "n": n,
        "media": media,
        "varianza": varianza,
        "desviacion": desviacion,
        "mediana": mediana,
        "mad": mad,
        "sigma_mad": 1.4826 * mad,
    }


def mapa_variabilidad_local(imagen, ventana=7):
    """Desvío estándar local en una ventana ventana x ventana."""
    img = np.asarray(imagen, dtype=float)
    media = ndi.uniform_filter(img, ventana)
    media_cuad = ndi.uniform_filter(img ** 2, ventana)
    return np.sqrt(np.clip(media_cuad - media ** 2, 0, None))


def diferencias_pixeles(imagen):
    """
    Diferencias entre píxeles vecinos (horizontal y vertical).
    Si el ruido es blanco con desvío sigma, Var(dx) = 2 sigma^2,
    así que sigma ~ std(dx)/sqrt(2); la versión MAD ignora los bordes.
    """
    img = np.asarray(imagen, dtype=float)
    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)
    todas = np.concatenate([dx.ravel(), dy.ravel()])

    mad = np.median(np.abs(todas - np.median(todas)))
    curtosis = np.mean((todas - todas.mean()) ** 4) / todas.var() ** 2

    return {
        "dx": dx,
        "dy": dy,
        "sigma_std": todas.std() / np.sqrt(2),
        "sigma_mad": 1.4826 * mad / np.sqrt(2),
        "curtosis": curtosis,
    }


def _distancia_radial(forma):
    alto, ancho = forma
    filas, columnas = np.indices(forma)
    return np.hypot(filas - alto // 2, columnas - ancho // 2)


def espectro_fourier(imagen):
    """
    Espectro de magnitud (log) centrado, con ventana de Hann para
    evitar la cruz que generan los bordes. Busca el pico más prominente
    fuera del centro respecto de la mediana de su anillo: un pico alto
    indica un patrón periódico (artefacto estructurado).
    """
    img = np.asarray(imagen, dtype=float)
    alto, ancho = img.shape
    ventana = np.outer(np.hanning(alto), np.hanning(ancho))

    magnitud = np.abs(np.fft.fftshift(np.fft.fft2((img - img.mean()) * ventana)))
    log_mag = np.log1p(magnitud)

    radio = _distancia_radial(img.shape)
    anillos = radio.astype(int)
    mediana_anillo = np.asarray(
        ndi.median(log_mag, anillos, np.arange(anillos.max() + 1))
    )
    prominencia = log_mag - mediana_anillo[anillos]

    radio_min = max(4, int(0.015 * min(alto, ancho)))
    candidatos = np.where(radio > radio_min, prominencia, -np.inf)
    fila, columna = np.unravel_index(np.argmax(candidatos), candidatos.shape)

    fy = (fila - alto // 2) / alto        # ciclos / píxel
    fx = (columna - ancho // 2) / ancho
    frecuencia = np.hypot(fx, fy)

    # Densidad espectral radial (sin ventana) para el análisis por bandas
    potencia = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
    radio_norm = radio / (min(alto, ancho) / 2)   # 1 = Nyquist

    def energia(lo, hi):
        return potencia[(radio_norm > lo) & (radio_norm <= hi)].sum()

    relacion_altas = energia(0.25, 0.5) / max(energia(0.10, 0.25), 1e-12)

    n_bins = min(alto, ancho) // 2
    psd_radial = np.asarray(
        ndi.mean(potencia, anillos, np.arange(n_bins))
    )

    return {
        "log_magnitud": log_mag,
        "pico_prominencia": float(prominencia[fila, columna]),
        "pico_posicion": (int(fila), int(columna)),
        "pico_frecuencia": (float(fy), float(fx)),
        "pico_periodo": 1.0 / frecuencia if frecuencia > 0 else np.inf,
        "relacion_altas": float(relacion_altas),
        "psd_radial": psd_radial,
        "frecuencias_radiales": np.arange(n_bins) / min(alto, ancho),
    }


def autocorrelacion(imagen):
    """
    Autocorrelación normalizada (Wiener-Khinchin: IFFT de |FFT|^2),
    centrada en el origen. Ruido blanco -> cae casi a cero en lag 1;
    desenfoque -> pico ancho; patrón periódico -> oscilaciones.
    """
    img = np.asarray(imagen, dtype=float)
    centrada = img - img.mean()
    potencia = np.abs(np.fft.fft2(centrada)) ** 2
    ac = np.real(np.fft.ifft2(potencia))
    ac /= ac[0, 0]
    ac = np.fft.fftshift(ac)

    alto, ancho = img.shape
    fila_c, col_c = alto // 2, ancho // 2
    perfil_x = ac[fila_c, col_c:]
    perfil_y = ac[fila_c:, col_c]

    def ancho_medio(perfil):
        # primer lag donde la autocorrelación cae por debajo de 0.5
        debajo = np.where(perfil < 0.5)[0]
        return int(debajo[0]) if debajo.size else len(perfil)

    return {
        "mapa": ac,
        "lag1_x": float(perfil_x[1]),
        "lag1_y": float(perfil_y[1]),
        "perfil_x": perfil_x,
        "perfil_y": perfil_y,
        "ancho_medio_x": ancho_medio(perfil_x),
        "ancho_medio_y": ancho_medio(perfil_y),
    }


def descomposicion_wavelet(imagen, wavelet="db2", niveles=3):
    """
    Descomposición wavelet 2D. El ruido estimado con el detalle
    diagonal del nivel 1: sigma = mediana(|HH1|) / 0.6745 (Donoho).
    """
    img = np.asarray(imagen, dtype=float)
    max_niveles = pywt.dwtn_max_level(img.shape, wavelet)
    niveles = int(max(1, min(niveles, max_niveles)))

    coeficientes = pywt.wavedec2(img, wavelet, level=niveles)
    horizontal_1, vertical_1, diagonal_1 = coeficientes[-1]

    # energía de detalle por nivel (1 = más fino)
    energias = []
    for detalles in coeficientes[:0:-1]:
        energias.append(sum(float(np.mean(d ** 2)) for d in detalles))

    return {
        "niveles": niveles,
        "aproximacion": coeficientes[0],
        "horizontal_1": horizontal_1,
        "vertical_1": vertical_1,
        "diagonal_1": diagonal_1,
        "sigma_donoho": float(np.median(np.abs(diagonal_1)) / 0.6745),
        "energias_detalle": energias,
    }


def indicador_fondo(imagen):
    """
    Compara el nivel medio de las cuatro esquinas contra la zona
    central (suavizado leve). Si difiere mucho, hay inhomogeneidad
    (viñeteado, iluminación no uniforme). El mapa de fondo que se
    devuelve es un gaussiano de gran escala, para inspección visual.
    """
    img = np.asarray(imagen, dtype=float)
    fondo = ndi.gaussian_filter(img, sigma=max(3, min(img.shape) / 10))
    suave = ndi.gaussian_filter(img, sigma=3)

    alto, ancho = img.shape
    lado = max(2, int(0.12 * min(alto, ancho)))
    esquinas = np.mean([
        suave[:lado, :lado].mean(),
        suave[:lado, -lado:].mean(),
        suave[-lado:, :lado].mean(),
        suave[-lado:, -lado:].mean(),
    ])

    fc, cc = alto // 2, ancho // 2
    medio = max(1, int(0.2 * min(alto, ancho)))
    centro = suave[fc - medio:fc + medio, cc - medio:cc + medio]

    return {
        "fondo": fondo,
        "relacion_borde_centro": float(esquinas / max(centro.mean(), 1e-12)),
    }


def fraccion_impulsos(imagen, umbral=60):
    """
    Fracción de píxeles que se apartan mucho de la mediana 3x3:
    detecta ruido impulsivo (sal y pimienta).
    """
    img = np.asarray(imagen, dtype=float)
    return float(np.mean(np.abs(img - ndi.median_filter(img, 3)) > umbral))


def diagnosticar(imagen, ventana=7, wavelet="db2", niveles=3, **kwargs):
    """
    Corre todos los análisis de la Actividad 1 sobre una imagen 2D
    (ya en escala de grises) y devuelve un diccionario con los
    indicadores y los mapas. No interpreta: la hipótesis de la
    degradación dominante se formula en el informe.
    """
    img = a_escala_de_grises(imagen)
    if min(img.shape) < 16:
        raise ValueError("La región es muy chica: usá al menos 16x16 píxeles.")

    estadisticas = estadisticas_robustas(img)
    variabilidad = mapa_variabilidad_local(img, ventana)
    diferencias = diferencias_pixeles(img)
    fourier = espectro_fourier(img)
    auto = autocorrelacion(img)
    ondita = descomposicion_wavelet(img, wavelet, niveles)
    fondo = indicador_fondo(img)

    rango = np.percentile(img, 99) - np.percentile(img, 1)

    indicadores = {
        "rango_p1_p99": rango,
        "sigma_wavelet": ondita["sigma_donoho"],
        "sigma_wavelet_sobre_rango": ondita["sigma_donoho"] / max(rango, 1e-12),
        "sigma_diferencias_mad": diferencias["sigma_mad"],
        "curtosis_diferencias": diferencias["curtosis"],
        "fraccion_impulsos": fraccion_impulsos(img),
        "pico_fft_prominencia": fourier["pico_prominencia"],
        "pico_fft_periodo_px": fourier["pico_periodo"],
        "relacion_altas_medias": fourier["relacion_altas"],
        "autocorr_lag1_x": auto["lag1_x"],
        "autocorr_lag1_y": auto["lag1_y"],
        "autocorr_ancho_medio_x": auto["ancho_medio_x"],
        "autocorr_ancho_medio_y": auto["ancho_medio_y"],
        "fondo_borde_centro": fondo["relacion_borde_centro"],
        "variabilidad_local_media": float(variabilidad.mean()),
    }

    return {
        "imagen": img,
        "rango": rango,
        "estadisticas": estadisticas,
        "variabilidad": variabilidad,
        "diferencias": diferencias,
        "fourier": fourier,
        "autocorrelacion": auto,
        "wavelet": ondita,
        "fondo": fondo,
        "indicadores": indicadores,
    }


# ------------------------------------------------------------
# Actividad 2 - Estimación de ruido
# (ROI homogénea, múltiples ROIs, MAD, diferencias, wavelet)
# ------------------------------------------------------------
METODOS_RUIDO = [
    "ROI homogénea",
    "Múltiples ROIs",
    "MAD",
    "Diferencias entre adquisiciones",
    "Diferencias locales",
    "Coeficientes wavelet",
]

def _valores(datos):
    valores = np.asarray(datos, dtype=float).ravel()
    return valores[np.isfinite(valores)]


def ruido_roi_homogenea(datos, media_fija=None):
    """
    σ = desvío muestral de una ROI homogénea.
    Con media fija μ0 (por ejemplo 0 en una ROI de fondo) se calcula además
    la dispersión alrededor de μ0: σ_μ0² = mean((x - μ0)²), que al conocer la
    media se divide por n. Vale σ_μ0² ≈ σ² + Δ², con Δ = media_obs - μ0:
    separa la parte aleatoria (σ) del sesgo sistemático (Δ).
    """
    est = estadisticas_robustas(datos)
    valores = _valores(datos)
    sigma = est["desviacion"]

    resultado = {
        "sigma": sigma,
        "media": est["media"],
        "n": est["n"],
        "snr": est["media"] / sigma if sigma > 0 else np.inf,
        "muestra": valores,
        "centro": est["media"],
    }
    if media_fija is not None:
        resultado["media_fija"] = float(media_fija)
        resultado["delta"] = est["media"] - media_fija
        resultado["sigma_media_fija"] = float(
            np.sqrt(np.mean((valores - media_fija) ** 2))
        )
    return resultado


def ruido_mad(datos, media_fija=None):
    """
    σ robusto = 1.4826 · mediana(|x - mediana|).
    Con media fija μ0 se informa además 1.4826 · mediana(|x - μ0|), que
    mezcla el ruido con el sesgo Δ = mediana - μ0 (igual que en ROI homogénea).
    """
    valores = _valores(datos)
    if valores.size < 2:
        raise ValueError("La región tiene menos de dos píxeles.")
    mediana = float(np.median(valores))
    mad = float(np.median(np.abs(valores - mediana)))
    resultado = {
        "sigma": 1.4826 * mad,
        "mad": mad,
        "mediana": mediana,
        "n": valores.size,
        "muestra": valores,
        "centro": mediana,
    }
    if media_fija is not None:
        resultado["media_fija"] = float(media_fija)
        resultado["delta"] = mediana - media_fija
        resultado["sigma_media_fija"] = 1.4826 * float(
            np.median(np.abs(valores - media_fija))
        )
    return resultado


def ruido_multiples_rois(lista_rois):
    """
    lista_rois: [(indice, pixeles), ...]. Devuelve el σ combinado
    sqrt(Σ(n_i - 1) s_i² / Σ(n_i - 1)) y el ajuste lineal var = a + b·media,
    que indica si el ruido depende de la intensidad.
    """
    if len(lista_rois) < 2:
        raise ValueError("Dibujá al menos 2 ROIs homogéneas.")

    filas = []
    residuos = []
    for indice, datos in lista_rois:
        n, media, varianza, desvio = calcular_estadisticas(datos)
        filas.append((indice, n, media, varianza, desvio))
        residuos.append(_valores(datos) - media)

    n = np.array([f[1] for f in filas], dtype=float)
    medias = np.array([f[2] for f in filas])
    varianzas = np.array([f[3] for f in filas])
    desvios = np.array([f[4] for f in filas])

    sigma = float(np.sqrt(np.sum((n - 1) * varianzas) / np.sum(n - 1)))
    cv_sigma = float(desvios.std() / max(desvios.mean(), 1e-12))

    ajuste = None
    if len(filas) >= 3 and np.ptp(medias) > 0:
        b, a = np.polyfit(medias, varianzas, 1)
        prediccion = a + b * medias
        ss_tot = np.sum((varianzas - varianzas.mean()) ** 2)
        r2 = 1 - np.sum((varianzas - prediccion) ** 2) / max(ss_tot, 1e-12)
        ajuste = {"a": float(a), "b": float(b), "r2": float(r2)}

    return {
        "sigma": sigma,
        "rois": filas,
        "cv_sigma": cv_sigma,
        "ajuste": ajuste,
        "muestra": np.concatenate(residuos),
        "centro": 0.0,
    }


def ruido_adquisiciones(imagen_a, imagen_b, referencia_limpia=False):
    """
    σ a partir de la diferencia entre dos adquisiciones de la misma escena.
    Independientes con igual σ: σ = std(A-B)/√2. Si B es una referencia
    sin ruido: σ = std(A-B). Se informa también la versión MAD (robusta a
    bordes mal alineados) y el offset medio entre ambas.
    """
    a = np.asarray(imagen_a, dtype=float)
    b = np.asarray(imagen_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"Las adquisiciones tienen tamaños distintos: {a.shape} vs {b.shape}."
        )

    diferencia = a - b
    factor = 1.0 if referencia_limpia else np.sqrt(2)
    mediana = float(np.median(diferencia))
    mad = float(np.median(np.abs(diferencia - mediana)))

    if mad == 0 and diferencia.std() == 0:
        raise ValueError("Las dos imágenes son idénticas: elegí otra adquisición.")

    return {
        "sigma": 1.4826 * mad / factor,
        "sigma_std": float(diferencia.std(ddof=1) / factor),
        "offset": float(diferencia.mean()),
        "factor": float(factor),
        "correlacion": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
        "mapa": diferencia,
        "muestra": (diferencia.ravel() - mediana) / factor,
        "centro": 0.0,
    }


def ruido_diferencias_locales(imagen):
    """σ con diferencias entre vecinos: 1.4826·MAD(dx, dy)/√2."""
    dif = diferencias_pixeles(imagen)
    todas = np.concatenate([dif["dx"].ravel(), dif["dy"].ravel()])
    return {
        "sigma": dif["sigma_mad"],
        "sigma_std": dif["sigma_std"],
        "curtosis": dif["curtosis"],
        "mapa": dif["dx"],
        "muestra": (todas - np.median(todas)) / np.sqrt(2),
        "centro": 0.0,
    }


def ruido_wavelet(imagen, wavelet="db2"):
    """σ de Donoho sobre HH1 (wavelet ortonormal: HH1 ~ N(0, σ) si es ruido)."""
    ondita = descomposicion_wavelet(imagen, wavelet, niveles=1)
    return {
        "sigma": ondita["sigma_donoho"],
        "mapa": ondita["diagonal_1"],
        "muestra": ondita["diagonal_1"].ravel(),
        "centro": 0.0,
    }


def estimar_ruido(
    imagen,
    metodo="ROI homogénea",
    datos_roi=None,
    recorte_roi=None,
    lista_rois=None,
    segunda=None,
    referencia_limpia=False,
    media_fija=None,
    wavelet="db2",
    motivos=None,
):
    """
    Estima σ del ruido con la estrategia elegida (`metodo`) y, para comparar,
    con todas las demás que se puedan aplicar con los datos disponibles.

    imagen:      región rectangular 2D (diferencias locales / wavelet).
    datos_roi:   píxeles de la ROI seleccionada (ROI homogénea / MAD).
    recorte_roi: rectángulo de esa ROI (para chequear que sea homogénea).
    lista_rois:  [(indice, pixeles)] de todas las ROIs (Múltiples ROIs).
    segunda:     misma región en otra adquisición (Diferencias entre adq.).
    motivos:     {metodo: texto} con la razón por la que un dato falta.
    """
    img = a_escala_de_grises(imagen)
    motivos = motivos or {}

    def requerir(dato, nombre, mensaje):
        if dato is None:
            raise ValueError(motivos.get(nombre, mensaje))
        return dato

    calculadores = {
        "ROI homogénea": lambda: ruido_roi_homogenea(
            requerir(datos_roi, "ROI", "Seleccioná una ROI homogénea."),
            media_fija,
        ),
        "Múltiples ROIs": lambda: ruido_multiples_rois(lista_rois or []),
        "MAD": lambda: ruido_mad(
            requerir(datos_roi, "ROI", "Seleccioná una ROI homogénea."),
            media_fija,
        ),
        "Diferencias entre adquisiciones": lambda: ruido_adquisiciones(
            img,
            requerir(segunda, "segunda", "Elegí otra adquisición de la escena."),
            referencia_limpia,
        ),
        "Diferencias locales": lambda: ruido_diferencias_locales(img),
        "Coeficientes wavelet": lambda: ruido_wavelet(img, wavelet),
    }

    resultados = {}
    for nombre, calcular in calculadores.items():
        try:
            resultados[nombre] = calcular()
        except Exception as error:
            resultados[nombre] = {"error": str(error)}

    principal = resultados[metodo]
    if "error" in principal:
        raise ValueError(f"{metodo}: {principal['error']}")
    sigma = principal["sigma"]

    # --- Valores complementarios (sin interpretar)
    complementarios = {"fraccion_impulsos": fraccion_impulsos(img)}
    if recorte_roi is not None and min(recorte_roi.shape) >= 3:
        # σ por diferencias locales dentro de la misma ROI: comparándolo con
        # el σ de la ROI se ve cuánto aporta la estructura de la región.
        complementarios["sigma_dif_locales_en_roi"] = (
            diferencias_pixeles(recorte_roi)["sigma_mad"]
        )

    return {
        "metodo": metodo,
        "sigma": sigma,
        "principal": principal,
        "resultados": resultados,
        "complementarios": complementarios,
    }


# ------------------------------------------------------------
# Actividad 3 - Reducción de ruido
# (gaussiano, bilateral, difusión anisotrópica, NLM, TV,
#  wavelet thresholding, Wiener local, BM3D)
# ------------------------------------------------------------
#
# Convención de unidades: todo parámetro de INTENSIDAD (σ del ruido,
# sigma_color, h, kappa) se da en la escala original de la imagen
# (0-255 en uint8), la misma del σ estimado en la Act. 2. Internamente
# se normaliza a [0, 1] para las funciones que lo necesitan.

METODOS_FILTRO = [
    "Gaussiano",
    "Bilateral",
    "Difusión anisotrópica",
    "Non-Local Means",
    "Variación Total",
    "Wavelet thresholding",
    "Wiener local",
    "BM3D",
]


def filtro_gaussiano(img, sigma_espacial=1.0):
    return ndi.gaussian_filter(img, sigma=sigma_espacial)


def filtro_bilateral(img, sigma_color=20.0, sigma_espacial=2.0):
    """
    Promedia vecinos cercanos en el espacio (sigma_espacial, px) Y en
    intensidad (sigma_color): no mezcla píxeles a ambos lados de un borde.
    """
    return cv2.bilateralFilter(
        img.astype(np.float32), d=-1,
        sigmaColor=float(sigma_color), sigmaSpace=float(sigma_espacial),
    ).astype(float)


def difusion_anisotropica(img, iteraciones=15, kappa=20.0, gamma=0.2,
                          funcion="exponencial"):
    """
    Perona-Malik. En cada iteración difunde hacia los 4 vecinos con un
    coeficiente g(|∇I|) que se apaga en los bordes (gradiente >> kappa):
      exponencial: g = exp(-(∇I/kappa)²)   (favorece bordes de alto contraste)
      cuadrática:  g = 1 / (1 + (∇I/kappa)²) (favorece regiones amplias)
    gamma <= 0.25 para que el esquema explícito sea estable.
    """
    if not 0 < gamma <= 0.25:
        raise ValueError("gamma debe estar en (0, 0.25] para que sea estable.")
    if kappa <= 0:
        raise ValueError("kappa debe ser positivo.")

    u = np.asarray(img, dtype=float).copy()
    for _ in range(int(iteraciones)):
        p = np.pad(u, 1, mode="edge")
        gradientes = (
            p[:-2, 1:-1] - u,   # norte
            p[2:, 1:-1] - u,    # sur
            p[1:-1, 2:] - u,    # este
            p[1:-1, :-2] - u,   # oeste
        )
        flujo = np.zeros_like(u)
        for d in gradientes:
            if funcion == "exponencial":
                g = np.exp(-((d / kappa) ** 2))
            else:
                g = 1.0 / (1.0 + (d / kappa) ** 2)
            flujo += g * d
        u += gamma * flujo
    return u


def filtro_nlm(img, h=10.0, sigma_ruido=0.0, tamano_parche=5,
               distancia_busqueda=6, escala=255.0):
    """
    Non-Local Means: cada píxel es el promedio de píxeles cuyo PARCHE
    (vecindario) se parece al suyo, buscando en una ventana amplia.
    h controla cuánto se parecen tienen que ser (unidades de intensidad).
    sigma_ruido > 0: se descuenta la varianza del ruido de la distancia
    entre parches (recomendado por skimage); 0 = no se usa.
    """
    salida = denoise_nl_means(
        img / escala,
        h=h / escala,
        sigma=sigma_ruido / escala,
        patch_size=int(tamano_parche),
        patch_distance=int(distancia_busqueda),
        fast_mode=True,
    )
    return salida * escala


def filtro_tv(img, peso=0.05, escala=255.0):
    """
    Variación Total (Chambolle): minimiza ||u - f||² + peso·TV(u).
    Produce regiones planas con bordes nítidos; mucho peso -> "cartoon".
    El peso es adimensional (imagen normalizada a [0, 1]).
    """
    return denoise_tv_chambolle(img / escala, weight=peso) * escala


def filtro_wavelet(img, sigma_ruido=0.0, wavelet="db2", metodo="BayesShrink",
                   modo="soft", niveles=0, escala=255.0):
    """
    Umbraliza los coeficientes de detalle wavelet (el ruido se reparte en
    muchos coeficientes chicos; la estructura, en pocos grandes).
      VisuShrink: umbral universal σ·√(2 ln N) (conservador, suaviza más).
      BayesShrink: un umbral por subbanda, adaptado a su varianza.
    sigma_ruido = 0: lo estima skimage (Donoho sobre HH1). niveles = 0: auto.
    """
    salida = denoise_wavelet(
        img / escala,
        sigma=(sigma_ruido / escala) if sigma_ruido > 0 else None,
        wavelet=wavelet,
        mode=modo,
        method=metodo,
        wavelet_levels=int(niveles) if niveles > 0 else None,
        rescale_sigma=True,
    )
    return salida * escala


def filtro_wiener_local(img, ventana=5, sigma_ruido=0.0):
    """
    Wiener adaptativo (Lee): en cada ventana
      u = μ + max(σ²_local - σ²_n, 0) / σ²_local · (f - μ)
    Donde la varianza local ≈ σ²_n (zona plana) promedia; donde es mucho
    mayor (bordes) deja la imagen casi igual.
    sigma_ruido = 0: σ²_n = promedio de las varianzas locales (scipy).
    """
    ruido = sigma_ruido ** 2 if sigma_ruido > 0 else None
    return wiener(img, mysize=int(ventana), noise=ruido)


def filtro_bm3d(img, sigma_ruido, etapas="Completo (2 etapas)", escala=255.0):
    """
    BM3D: agrupa parches parecidos en pilas 3D, filtra cada pila en un
    dominio transformado (umbral duro y luego Wiener) y reagrega.
    Necesita el σ del ruido (unidades de intensidad).
    """
    try:
        import bm3d
    except ImportError as error:
        raise ImportError(
            "Falta el paquete bm3d: correr 'pixi install' para actualizar el entorno."
        ) from error
    if sigma_ruido <= 0:
        raise ValueError("BM3D necesita σ del ruido > 0 (estimalo en la Act. 2).")

    etapa = (
        bm3d.BM3DStages.ALL_STAGES
        if etapas.startswith("Completo")
        else bm3d.BM3DStages.HARD_THRESHOLDING
    )
    salida = bm3d.bm3d(img / escala, sigma_psd=sigma_ruido / escala, stage_arg=etapa)
    return np.asarray(salida, dtype=float) * escala


def reducir_ruido(imagen, metodo="Gaussiano", escala=255.0, **p):
    """
    Aplica el filtro `metodo` a una imagen 2D y devuelve la imagen filtrada
    (float, misma escala). `p` son los parámetros del widget; cada método
    toma solo los suyos.
    """
    img = a_escala_de_grises(imagen)

    if metodo == "Gaussiano":
        return filtro_gaussiano(img, p["sigma_espacial"])
    if metodo == "Bilateral":
        return filtro_bilateral(img, p["sigma_color"], p["sigma_espacial"])
    if metodo == "Difusión anisotrópica":
        return difusion_anisotropica(
            img, p["iteraciones"], p["kappa"], p["gamma"], p["funcion_conduccion"]
        )
    if metodo == "Non-Local Means":
        return filtro_nlm(
            img, p["h"], p["sigma_ruido"], p["tamano_parche"],
            p["distancia_busqueda"], escala,
        )
    if metodo == "Variación Total":
        return filtro_tv(img, p["peso_tv"], escala)
    if metodo == "Wavelet thresholding":
        return filtro_wavelet(
            img, p["sigma_ruido"], p["wavelet"], p["umbral"],
            p["modo_umbral"], p["niveles_wavelet"], escala,
        )
    if metodo == "Wiener local":
        return filtro_wiener_local(img, p["ventana"], p["sigma_ruido"])
    if metodo == "BM3D":
        return filtro_bm3d(img, p["sigma_ruido"], p["etapas_bm3d"], escala)

    raise ValueError(f"Método desconocido: {metodo}")


# ------------------------------------------------------------
# Actividad 4 - Restauración y artefactos
# (deconvolución Wiener / Richardson-Lucy; filtro notch en Fourier)
# ------------------------------------------------------------
#
# 4.a  PSF y deconvolución
#      La PSF puede sintetizarse (gaussiana, disco, movimiento) o venir
#      de una capa de napari (PSF medida o provista por la cátedra).

TIPOS_PSF = ["Gaussiana", "Disco (desenfoque)", "Movimiento lineal", "Capa de napari"]
METODOS_DECONVOLUCION = ["Wiener", "Wiener no supervisado", "Richardson-Lucy"]


def psf_gaussiana(sigma=2.0):
    """PSF gaussiana isótropa de desvío sigma (px)."""
    if sigma <= 0:
        raise ValueError("El sigma de la PSF debe ser positivo.")
    radio = int(np.ceil(3 * sigma))
    y, x = np.mgrid[-radio:radio + 1, -radio:radio + 1]
    psf = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def psf_disco(radio=3.0):
    """PSF de disco (desenfoque por apertura circular)."""
    if radio <= 0:
        raise ValueError("El radio de la PSF debe ser positivo.")
    r = int(np.ceil(radio))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    psf = (np.hypot(x, y) <= radio).astype(float)
    return psf / psf.sum()


def psf_movimiento(longitud=9.0, angulo=0.0):
    """PSF de movimiento lineal: un segmento de `longitud` px a `angulo` grados."""
    if longitud < 2:
        raise ValueError("La longitud de la PSF debe ser de al menos 2 px.")
    radio = int(np.ceil(longitud / 2))
    psf = np.zeros((2 * radio + 1, 2 * radio + 1))
    theta = np.deg2rad(angulo)
    pasos = np.linspace(-longitud / 2, longitud / 2, int(4 * longitud))
    filas = np.round(radio - pasos * np.sin(theta)).astype(int)
    columnas = np.round(radio + pasos * np.cos(theta)).astype(int)
    np.add.at(psf, (filas, columnas), 1.0)
    return psf / psf.sum()


def psf_desde_datos(datos):
    """PSF a partir de una capa: pasa a gris, recorta negativos y normaliza."""
    psf = a_escala_de_grises(datos)
    psf = np.clip(psf - psf.min(), 0, None)
    total = psf.sum()
    if total <= 0:
        raise ValueError("La capa elegida como PSF está vacía.")
    return psf / total


def construir_psf(tipo, sigma_psf=2.0, radio_psf=3.0, longitud_psf=9.0,
                  angulo_psf=0.0, datos_psf=None):
    if tipo == "Gaussiana":
        return psf_gaussiana(sigma_psf)
    if tipo == "Disco (desenfoque)":
        return psf_disco(radio_psf)
    if tipo == "Movimiento lineal":
        return psf_movimiento(longitud_psf, angulo_psf)
    if tipo == "Capa de napari":
        if datos_psf is None:
            raise ValueError("Elegí la capa que contiene la PSF.")
        return psf_desde_datos(datos_psf)
    raise ValueError(f"Tipo de PSF desconocido: {tipo}")


def _con_borde(img, ancho):
    """Extiende los bordes (evita el anillado de la convolución circular)."""
    return np.pad(img, ancho, mode="edge")


def _sin_borde(img, ancho):
    return img[ancho:-ancho, ancho:-ancho] if ancho > 0 else img


def deconvolucion_wiener(img, psf, balance=0.01, escala=255.0, no_supervisado=False):
    """
    Deconvolución de Wiener: U = (H* / (|H|² + balance·|R|²)) · F.
    `balance` es la regularización: es el peso del término que penaliza las
    altas frecuencias. Un valor del orden de NSR = σ²/Var(imagen) equilibra
    resolución y ruido; más chico afina pero amplifica el ruido.
    "No supervisado" estima esa regularización con un muestreo bayesiano
    (skimage.restoration.unsupervised_wiener) en vez de fijarla a mano.
    """
    ancho = max(psf.shape) // 2 + 1
    extendida = _con_borde(np.asarray(img, dtype=float) / escala, ancho)

    if no_supervisado:
        salida, _ = unsupervised_wiener(extendida, psf, clip=False)
        usado = None
    else:
        if balance <= 0:
            raise ValueError("El balance (regularización) debe ser positivo.")
        salida = wiener_deconv(extendida, psf, balance=balance, clip=False)
        usado = balance

    return _sin_borde(np.asarray(salida, dtype=float), ancho) * escala, usado


def richardson_lucy(img, psf, iteraciones=20, tolerancia=0.0, escala=255.0):
    """
    Richardson-Lucy: iteración multiplicativa que supone ruido de Poisson.
      u_{k+1} = u_k · [ (f / (u_k * h)) * h⁻ ]
    Devuelve (imagen, historial). En cada iteración se registra:
      - residuo_rms: RMS de f - (u_k * h), el ajuste a los datos observados;
      - cambio: ||u_k - u_{k-1}|| / ||u_{k-1}||, cuánto se mueve la solución.
    RL no converge a algo estable: amplifica el ruido a partir de cierta
    iteración, así que hace falta un criterio de parada. Con tolerancia > 0
    se corta cuando el cambio relativo baja de ese valor.
    """
    psf = np.asarray(psf, dtype=float)
    psf_espejo = psf[::-1, ::-1]
    ancho = max(psf.shape) // 2 + 1
    f = _con_borde(np.asarray(img, dtype=float) / escala, ancho)

    u = np.full(f.shape, f.mean())
    historial = []
    eps = 1e-12

    for k in range(1, int(iteraciones) + 1):
        convolucion = fftconvolve(u, psf, mode="same")
        u_previo = u
        u = u * fftconvolve(f / (convolucion + eps), psf_espejo, mode="same")
        u = np.clip(u, 0, None)

        cambio = float(
            np.linalg.norm(u - u_previo) / max(np.linalg.norm(u_previo), eps)
        )
        historial.append({
            "iteracion": k,
            "residuo_rms": float(np.sqrt(np.mean((f - convolucion) ** 2)) * escala),
            "cambio": cambio,
        })
        if tolerancia > 0 and cambio < tolerancia:
            break

    return _sin_borde(u, ancho) * escala, historial


def restaurar(imagen, metodo="Wiener", psf=None, balance=0.01, iteraciones=20,
              tolerancia=0.0, escala=255.0):
    """
    Deconvolución de una imagen 2D con una PSF dada.
    Devuelve (imagen_restaurada, informacion) donde `informacion` trae el
    balance usado (Wiener) o el historial de iteraciones (Richardson-Lucy).
    """
    img = a_escala_de_grises(imagen)
    if psf is None:
        raise ValueError("Hace falta una PSF.")

    if metodo in ("Wiener", "Wiener no supervisado"):
        salida, usado = deconvolucion_wiener(
            img, psf, balance, escala, no_supervisado=(metodo != "Wiener")
        )
        return salida, {"balance_usado": usado}
    if metodo == "Richardson-Lucy":
        salida, historial = richardson_lucy(img, psf, iteraciones, tolerancia, escala)
        return salida, {"historial": historial}

    raise ValueError(f"Método de deconvolución desconocido: {metodo}")


# 4.b  Artefactos periódicos: detección en Fourier y filtro notch

TIPOS_NOTCH = ["Gaussiano", "Ideal", "Butterworth"]


def detectar_picos_fft(imagen, cantidad=6, radio_min_rel=0.015, separacion=5):
    """
    Picos del espectro que sobresalen de su anillo (patrones periódicos).
    Para cada uno devuelve su posición en el espectro centrado, la
    frecuencia (ciclos/px), el período (px), la orientación y la
    prominencia sobre la mediana del anillo. Los pares simétricos
    (f y -f) se informan una sola vez.
    """
    img = a_escala_de_grises(imagen)
    alto, ancho = img.shape
    ventana = np.outer(np.hanning(alto), np.hanning(ancho))
    log_mag = np.log1p(
        np.abs(np.fft.fftshift(np.fft.fft2((img - img.mean()) * ventana)))
    )

    radio = _distancia_radial(img.shape)
    anillos = radio.astype(int)
    mediana_anillo = np.asarray(
        ndi.median(log_mag, anillos, np.arange(anillos.max() + 1))
    )
    prominencia = log_mag - mediana_anillo[anillos]

    maximos = ndi.maximum_filter(prominencia, size=separacion)
    radio_min = max(4, int(radio_min_rel * min(alto, ancho)))
    candidatos = (prominencia >= maximos) & (radio > radio_min)

    filas, columnas = np.where(candidatos)
    orden = np.argsort(prominencia[filas, columnas])[::-1]

    centro_f, centro_c = alto // 2, ancho // 2
    picos = []
    vistos = set()
    for indice in orden:
        fila, columna = int(filas[indice]), int(columnas[indice])
        simetrico = (2 * centro_f - fila, 2 * centro_c - columna)
        if (fila, columna) in vistos or simetrico in vistos:
            continue
        vistos.add((fila, columna))

        fy = (fila - centro_f) / alto
        fx = (columna - centro_c) / ancho
        frecuencia = np.hypot(fx, fy)
        picos.append({
            "fila": fila,
            "columna": columna,
            "fy": float(fy),
            "fx": float(fx),
            "frecuencia": float(frecuencia),
            "periodo_px": float(1 / frecuencia) if frecuencia > 0 else np.inf,
            "angulo_grados": float(np.degrees(np.arctan2(fy, fx))),
            "prominencia": float(prominencia[fila, columna]),
        })
        if len(picos) >= cantidad:
            break

    return picos, log_mag


def mascara_notch(forma, picos, radio=5.0, tipo="Gaussiano", orden=2):
    """
    Máscara multiplicativa (espectro centrado) que atenúa cada pico y su
    simétrico. `radio` es la extensión del notch en píxeles del espectro:
    tiene que cubrir el ancho del pico, pero no más.
      Ideal: corta de golpe (puede dejar ondulaciones tipo Gibbs).
      Gaussiano: transición suave.
      Butterworth: transición controlada por el orden.
    """
    if radio <= 0:
        raise ValueError("El radio del notch debe ser positivo.")

    alto, ancho = forma
    centro_f, centro_c = alto // 2, ancho // 2
    filas, columnas = np.indices(forma)
    mascara = np.ones(forma, dtype=float)

    for pico in picos:
        posiciones = [
            (pico["fila"], pico["columna"]),
            (2 * centro_f - pico["fila"], 2 * centro_c - pico["columna"]),
        ]
        for fila, columna in posiciones:
            d = np.hypot(filas - fila, columnas - columna)
            if tipo == "Ideal":
                mascara[d <= radio] = 0.0
            elif tipo == "Butterworth":
                with np.errstate(divide="ignore"):
                    mascara *= 1.0 / (1.0 + (radio / np.maximum(d, 1e-9)) ** (2 * orden))
            else:
                mascara *= 1.0 - np.exp(-(d ** 2) / (2 * radio ** 2))

    return mascara


def aplicar_notch(imagen, mascara):
    """Filtra la imagen con la máscara en el espectro centrado."""
    img = a_escala_de_grises(imagen)
    espectro = np.fft.fftshift(np.fft.fft2(img))
    filtrada = np.real(np.fft.ifft2(np.fft.ifftshift(espectro * mascara)))
    return filtrada, espectro


def verificar_notch(original, filtrada):
    """
    Valores para controlar que el notch se haya llevado el patrón y no la
    anatomía (la lectura es de quien analiza):
      - fraccion_energia_removida: ||o - f||² / ||o - media||².
      - correlacion_residuo_filtrada: correlación entre lo removido y lo
        que quedó (la anatomía estimada). El patrón periódico es ajeno a
        la anatomía: cerca de 0 significa que lo removido es independiente
        de lo que se conserva.
      - correlacion_residuo_original: lo mismo contra la imagen de entrada,
        que todavía contiene el patrón.
      - razon_energia_gradiente: energía de gradiente después / antes
        (cuánto se ablandaron los bordes).
      - razon_varianza: varianza después / antes.
    """
    original = np.asarray(original, dtype=float)
    filtrada = np.asarray(filtrada, dtype=float)
    residuo = original - filtrada

    def energia_gradiente(img):
        gy, gx = np.gradient(img)
        return float(np.mean(gx ** 2 + gy ** 2))

    energia_total = float(np.sum((original - original.mean()) ** 2))

    def correlacion(otra):
        if residuo.std() == 0 or otra.std() == 0:
            return 0.0
        return float(np.corrcoef(residuo.ravel(), otra.ravel())[0, 1])

    return {
        "fraccion_energia_removida": float(np.sum(residuo ** 2)) / max(energia_total, 1e-12),
        "correlacion_residuo_filtrada": correlacion(filtrada),
        "correlacion_residuo_original": correlacion(original),
        "razon_energia_gradiente": energia_gradiente(filtrada)
        / max(energia_gradiente(original), 1e-12),
        "razon_varianza": float(filtrada.var()) / max(float(original.var()), 1e-12),
        "residuo_amplitud_pico_a_pico": float(np.ptp(residuo)),
    }


# ------------------------------------------------------------
# Actividad 5 - Corrección de fondo
# (dark/flat-field, kernel de gran escala, rolling ball,
#  white top-hat, homomórfica)
# ------------------------------------------------------------
#
# Cada función devuelve (corregida, fondo_estimado): el fondo se puede
# mirar como capa para ver qué se está quitando.

METODOS_FONDO = [
    "Dark-field / flat-field",
    "Kernel de gran escala",
    "Rolling ball",
    "White top-hat",
    "Homomórfica",
]


def _nivel(img, fondo, conservar_nivel):
    """Offset que se suma tras restar el fondo (para no dejar todo en ~0)."""
    return float(np.mean(fondo)) if conservar_nivel else 0.0


def fondo_flat_field(img, flat, dark=None):
    """
    Corrección por campos de referencia:
        corregida = (I - D) / (F - D) · media(F - D)
    D (dark-field) es la señal con el obturador cerrado: corriente de
    oscuridad y offset del sensor. F (flat-field) es una toma de campo
    uniforme: recoge la respuesta del sistema (viñeteo, polvo, ganancia
    de cada píxel). El factor media(F - D) devuelve la imagen a su
    nivel original en vez de dejarla alrededor de 1.
    """
    flat = a_escala_de_grises(flat)
    oscuro = np.zeros_like(img) if dark is None else a_escala_de_grises(dark)
    if flat.shape != img.shape or oscuro.shape != img.shape:
        raise ValueError(
            "El flat-field y el dark-field deben tener el mismo tamaño que la región."
        )

    ganancia = flat - oscuro
    media = float(np.mean(ganancia))
    if media <= 0:
        raise ValueError("El flat-field menos el dark-field da un campo vacío.")

    corregida = (img - oscuro) / np.where(np.abs(ganancia) < 1e-9, 1e-9, ganancia) * media
    return corregida, ganancia


def fondo_kernel_grande(img, sigma=50.0, tipo="Gaussiano", modo="Resta",
                        conservar_nivel=True):
    """
    Estima el fondo suavizando la imagen con un kernel mucho más grande
    que las estructuras de interés (gaussiano o mediana) y lo quita.
      Resta:    corregida = I - B + media(B)   (fondo aditivo)
      División: corregida = I / B · media(B)   (fondo multiplicativo)
    """
    if sigma <= 0:
        raise ValueError("El tamaño del kernel debe ser positivo.")

    if tipo == "Mediana":
        fondo = ndi.median_filter(img, size=int(max(3, 2 * round(sigma) + 1)))
    else:
        fondo = ndi.gaussian_filter(img, sigma=sigma)

    if modo == "División":
        seguro = np.where(np.abs(fondo) < 1e-9, 1e-9, fondo)
        return img / seguro * float(np.mean(fondo)), fondo
    return img - fondo + _nivel(img, fondo, conservar_nivel), fondo


def fondo_rolling_ball(img, radio=50.0, conservar_nivel=True):
    """
    Rolling ball: hace rodar una esfera de radio `radio` por debajo de la
    superficie de intensidades; lo que toca es el fondo. El radio tiene
    que ser mayor que las estructuras que se quieren conservar.
    """
    if radio <= 0:
        raise ValueError("El radio de la bola debe ser positivo.")
    fondo = rolling_ball(img, radius=float(radio))
    return img - fondo + _nivel(img, fondo, conservar_nivel), fondo


def fondo_tophat(img, radio=25.0, conservar_nivel=True):
    """
    White top-hat: I - apertura(I). La apertura con un disco de radio
    `radio` borra todo lo más chico que el disco (queda el fondo), así
    que la resta deja las estructuras claras y chicas.
    """
    if radio <= 0:
        raise ValueError("El radio del top-hat debe ser positivo.")
    elemento = disk(int(round(radio)))
    corregida = white_tophat(img, elemento)
    fondo = img - corregida
    return corregida + _nivel(img, fondo, conservar_nivel), fondo


def fondo_homomorfico(img, gamma_bajo=0.5, gamma_alto=1.5, corte=0.05):
    """
    Corrección homomórfica: si I = iluminación · reflectancia, en log se
    vuelve una suma. La iluminación es de baja frecuencia y la
    reflectancia de alta, así que se filtra en log con
        H = gamma_bajo + (gamma_alto - gamma_bajo)·(1 - exp(-D²/D0²))
    y se vuelve con exp. gamma_bajo < 1 atenúa la iluminación despareja;
    gamma_alto > 1 realza el detalle. `corte` es D0 en fracción de
    Nyquist: más grande = se considera "iluminación" un rango más amplio.
    """
    if not 0 < corte <= 1:
        raise ValueError("El corte debe estar entre 0 y 1 (fracción de Nyquist).")

    desplazado = img - img.min() + 1.0
    logaritmo = np.log(desplazado)

    alto, ancho = img.shape
    radio = _distancia_radial(img.shape) / (min(alto, ancho) / 2)
    filtro = gamma_bajo + (gamma_alto - gamma_bajo) * (
        1 - np.exp(-(radio ** 2) / (corte ** 2))
    )

    espectro = np.fft.fftshift(np.fft.fft2(logaritmo))
    corregida = np.exp(np.real(np.fft.ifft2(np.fft.ifftshift(espectro * filtro))))
    corregida = corregida + img.min() - 1.0

    # Atenuar las bajas frecuencias comprime el nivel general: se reescala
    # con una ganancia constante para volver a la media original (la
    # ganancia es un factor único, no cambia el contraste relativo).
    media_actual = float(np.mean(corregida))
    if abs(media_actual) > 1e-9:
        corregida = corregida * (float(np.mean(img)) / media_actual)

    # Fondo implícito: lo que se le quitó a la imagen
    return corregida, img - corregida + float(np.mean(img))


def corregir_fondo(imagen, metodo="Rolling ball", flat=None, dark=None,
                   sigma_fondo=50.0, tipo_kernel="Gaussiano",
                   modo_correccion="Resta", radio_bola=50.0, radio_tophat=25.0,
                   gamma_bajo=0.5, gamma_alto=1.5, corte=0.05,
                   conservar_nivel=True, **kwargs):
    """Aplica el método de corrección de fondo elegido: (corregida, fondo)."""
    img = a_escala_de_grises(imagen)

    if metodo == "Dark-field / flat-field":
        if flat is None:
            raise ValueError("Elegí la capa de flat-field.")
        return fondo_flat_field(img, flat, dark)
    if metodo == "Kernel de gran escala":
        return fondo_kernel_grande(
            img, sigma_fondo, tipo_kernel, modo_correccion, conservar_nivel
        )
    if metodo == "Rolling ball":
        return fondo_rolling_ball(img, radio_bola, conservar_nivel)
    if metodo == "White top-hat":
        return fondo_tophat(img, radio_tophat, conservar_nivel)
    if metodo == "Homomórfica":
        return fondo_homomorfico(img, gamma_bajo, gamma_alto, corte)

    raise ValueError(f"Método de corrección de fondo desconocido: {metodo}")


def medidas_fondo(img, lista_rois=None):
    """
    Valores para comparar antes y después (uniformidad del fondo).
      fondo_borde_centro: nivel medio de las esquinas / nivel del centro.
      rango_fondo_suave: cuánto varía la imagen muy suavizada (p95 - p5),
        es decir la amplitud de la inhomogeneidad que queda.
      cv_medias_rois: dispersión de las medias entre ROIs (si hay 2 o más);
        con el fondo corregido, ROIs del mismo tejido deberían acercarse.
    """
    img = np.asarray(img, dtype=float)
    suave = ndi.gaussian_filter(img, sigma=max(3, min(img.shape) / 20))

    medidas = {
        "media": float(img.mean()),
        "desvio": float(img.std()),
        "fondo_borde_centro": indicador_fondo(img)["relacion_borde_centro"],
        "rango_fondo_suave": float(np.percentile(suave, 95) - np.percentile(suave, 5)),
        "sigma_wavelet": descomposicion_wavelet(img, "db2", 1)["sigma_donoho"],
    }
    if lista_rois and len(lista_rois) >= 2:
        medias = np.array([float(np.mean(datos)) for _, datos in lista_rois])
        medidas["cv_medias_rois"] = float(medias.std() / max(abs(medias.mean()), 1e-12))
    return medidas


# ------------------------------------------------------------
# Actividad 6 - Realce
# (gamma, log, sigmoidal, CLAHE, pasa-altos, pasa-banda, unsharp)
# ------------------------------------------------------------
#
# Las transformaciones de intensidad (gamma, log, sigmoidal, CLAHE)
# trabajan sobre la imagen normalizada a [0, 1] y vuelven al rango
# original, así el antes y el después son comparables.

METODOS_REALCE = [
    "Gamma",
    "Logarítmica",
    "Sigmoidal",
    "CLAHE",
    "Pasa-altos",
    "Pasa-banda",
    "Unsharp masking",
]


def _normalizar(img):
    """Lleva la imagen a [0, 1] y devuelve (normalizada, minimo, rango)."""
    img = np.asarray(img, dtype=float)
    minimo = float(img.min())
    rango = float(img.max()) - minimo
    if rango <= 0:
        raise ValueError("La región es constante: no hay nada que realzar.")
    return (img - minimo) / rango, minimo, rango


def realce_gamma(img, gamma=0.7):
    """
    s = r^gamma sobre [0, 1].
    gamma < 1 aclara y expande el contraste de las zonas oscuras;
    gamma > 1 oscurece y expande el de las zonas claras.
    """
    if gamma <= 0:
        raise ValueError("gamma debe ser positivo.")
    normalizada, minimo, rango = _normalizar(img)
    return normalizada ** gamma * rango + minimo


def realce_logaritmico(img, factor=10.0):
    """
    s = log(1 + factor·r) / log(1 + factor).
    Expande los niveles oscuros y comprime los claros; `factor` regula
    cuánto (más grande = más expansión de las sombras).
    """
    if factor <= 0:
        raise ValueError("El factor debe ser positivo.")
    normalizada, minimo, rango = _normalizar(img)
    salida = np.log1p(factor * normalizada) / np.log1p(factor)
    return salida * rango + minimo


def realce_sigmoidal(img, centro=0.5, ganancia=10.0):
    """
    s = 1 / (1 + exp(ganancia·(centro - r))), reescalada a [0, 1].
    Estira el contraste alrededor de `centro` (en fracción del rango) y
    comprime los extremos. `ganancia` es la pendiente.
    """
    normalizada, minimo, rango = _normalizar(img)
    salida = 1.0 / (1.0 + np.exp(ganancia * (centro - normalizada)))
    bajo = 1.0 / (1.0 + np.exp(ganancia * centro))
    alto = 1.0 / (1.0 + np.exp(ganancia * (centro - 1.0)))
    salida = (salida - bajo) / max(alto - bajo, 1e-12)
    return salida * rango + minimo


def realce_clahe(img, tamano_mosaico=64, limite_contraste=0.01, niveles=256):
    """
    CLAHE: ecualiza el histograma por mosaicos e interpola entre ellos.
    `limite_contraste` recorta el histograma de cada mosaico antes de
    ecualizar: es el freno a la amplificación del ruido en zonas planas.
    """
    normalizada, minimo, rango = _normalizar(img)
    salida = equalize_adapthist(
        normalizada,
        kernel_size=int(max(3, tamano_mosaico)),
        clip_limit=float(limite_contraste),
        nbins=int(niveles),
    )
    return salida * rango + minimo


def realce_pasa_altos(img, sigma=3.0, conservar_nivel=True):
    """
    Pasa-altos espacial: I - gaussiana(I, sigma). Deja los detalles finos
    y bordes; quita las variaciones lentas (incluido el fondo).
    """
    img = np.asarray(img, dtype=float)
    detalle = img - ndi.gaussian_filter(img, sigma=sigma)
    return detalle + (float(img.mean()) if conservar_nivel else 0.0)


def realce_pasa_banda(img, sigma_bajo=1.0, sigma_alto=8.0, conservar_nivel=True):
    """
    Pasa-banda por diferencia de gaussianas: gauss(σ_bajo) - gauss(σ_alto).
    Conserva las estructuras de tamaño intermedio: σ_bajo frena el ruido
    de 1 px y σ_alto quita el fondo.
    """
    if sigma_bajo >= sigma_alto:
        raise ValueError("σ bajo debe ser menor que σ alto.")
    img = np.asarray(img, dtype=float)
    banda = ndi.gaussian_filter(img, sigma_bajo) - ndi.gaussian_filter(img, sigma_alto)
    return banda + (float(img.mean()) if conservar_nivel else 0.0)


def realce_unsharp(img, sigma=2.0, cantidad=1.0, umbral=0.0):
    """
    Unsharp masking: I + cantidad·(I - gaussiana(I, sigma)).
    Con `umbral` > 0 solo se realzan los detalles cuya amplitud supera
    ese valor (en intensidad): evita amplificar el ruido de las zonas planas.
    """
    img = np.asarray(img, dtype=float)
    detalle = img - ndi.gaussian_filter(img, sigma=sigma)
    if umbral > 0:
        detalle = np.where(np.abs(detalle) > umbral, detalle, 0.0)
    return img + cantidad * detalle


def realzar(imagen, metodo="Gamma", gamma=0.7, factor_log=10.0, centro=0.5,
            ganancia=10.0, tamano_mosaico=64, limite_contraste=0.01,
            sigma_altos=3.0, sigma_bajo=1.0, sigma_alto=8.0, sigma_unsharp=2.0,
            cantidad=1.0, umbral=0.0, conservar_nivel=True, **kwargs):
    """Aplica el método de realce elegido a una imagen 2D."""
    img = a_escala_de_grises(imagen)

    if metodo == "Gamma":
        return realce_gamma(img, gamma)
    if metodo == "Logarítmica":
        return realce_logaritmico(img, factor_log)
    if metodo == "Sigmoidal":
        return realce_sigmoidal(img, centro, ganancia)
    if metodo == "CLAHE":
        return realce_clahe(img, tamano_mosaico, limite_contraste)
    if metodo == "Pasa-altos":
        return realce_pasa_altos(img, sigma_altos, conservar_nivel)
    if metodo == "Pasa-banda":
        return realce_pasa_banda(img, sigma_bajo, sigma_alto, conservar_nivel)
    if metodo == "Unsharp masking":
        return realce_unsharp(img, sigma_unsharp, cantidad, umbral)

    raise ValueError(f"Método de realce desconocido: {metodo}")


def entropia(img, niveles=256):
    """
    Entropía de Shannon del histograma (bits). Baja cuando la
    transformación junta niveles distintos en uno solo, es decir cuando
    se pierde información de intensidad.
    """
    img = np.asarray(img, dtype=float)
    conteos, _ = np.histogram(img, bins=niveles)
    p = conteos[conteos > 0] / conteos.sum()
    return float(-np.sum(p * np.log2(p)))


def medidas_realce(img, lista_rois=None, limites=None):
    """
    Valores para evaluar un realce.
      contraste_rms / rango_p1_p99: cuánto contraste hay.
      entropia: información de intensidad que sobrevive.
      sigma_wavelet: nivel de ruido (si sube, el realce lo amplificó).
      fraccion_saturados: píxeles pegados a los extremos del rango
        `limites` (el de la imagen original): ahí la información se pierde
        porque distintos niveles quedan en el mismo valor.
      snr_roi_i = media/σ de cada ROI; cnr_i_j = |μi - μj| / √((σi²+σj²)/2).
    """
    img = np.asarray(img, dtype=float)
    medidas = {
        "media": float(img.mean()),
        "contraste_rms": float(img.std()),
        "rango_p1_p99": float(np.percentile(img, 99) - np.percentile(img, 1)),
        "entropia": entropia(img),
        "sigma_wavelet": descomposicion_wavelet(img, "db2", 1)["sigma_donoho"],
    }

    if limites is not None:
        bajo, alto = limites
        margen = 0.01 * (alto - bajo)
        medidas["fraccion_saturados"] = float(
            np.mean((img <= bajo + margen) | (img >= alto - margen))
        )

    if lista_rois:
        estadisticas = []
        for indice, datos in lista_rois:
            datos = np.asarray(datos, dtype=float)
            if datos.size < 2:
                continue
            media, desvio = float(datos.mean()), float(datos.std(ddof=1))
            estadisticas.append((indice, media, desvio))
            if desvio > 0:
                medidas[f"snr_roi{indice + 1}"] = media / desvio
        for (i, media_i, desvio_i), (j, media_j, desvio_j) in combinations(
            estadisticas, 2
        ):
            ruido = np.sqrt((desvio_i ** 2 + desvio_j ** 2) / 2)
            if ruido > 0:
                medidas[f"cnr_roi{i + 1}_roi{j + 1}"] = abs(media_i - media_j) / ruido

    return medidas


def comparar_realce(original, realzada):
    """
    Razones después/antes y correlación con la imagen de entrada.
    razon_sigma_wavelet > 1 significa que el ruido se amplificó;
    la correlación baja cuando el realce cambió mucho la estructura.
    """
    original = np.asarray(original, dtype=float)
    realzada = np.asarray(realzada, dtype=float)

    sigma_antes = descomposicion_wavelet(original, "db2", 1)["sigma_donoho"]
    sigma_despues = descomposicion_wavelet(realzada, "db2", 1)["sigma_donoho"]
    correlacion = 0.0
    if original.std() > 0 and realzada.std() > 0:
        correlacion = float(np.corrcoef(original.ravel(), realzada.ravel())[0, 1])

    return {
        "razon_sigma_wavelet": sigma_despues / max(sigma_antes, 1e-12),
        "razon_contraste_rms": float(realzada.std()) / max(float(original.std()), 1e-12),
        "razon_entropia": entropia(realzada) / max(entropia(original), 1e-12),
        "correlacion_con_original": correlacion,
    }


# ------------------------------------------------------------
# Actividad 7 - Evaluación
# (SIN referencia: SNR, CNR, uniformidad / CON referencia: MSE, PSNR, SSIM)
# ------------------------------------------------------------
#
# Las métricas sin referencia se calculan sobre las ROIs: SNR y
# uniformidad necesitan una región de un solo tejido, y CNR necesita
# dos regiones distintas. Las métricas con referencia comparan píxel a
# píxel contra una imagen de referencia del mismo tamaño.


def uniformidad(datos):
    """
    Uniformidad integral: 1 - (p95 - p5) / (p95 + p5).
    Vale 1 si la región es perfectamente plana y baja cuando hay
    gradiente de fondo. Se usan percentiles para que un píxel aislado
    no defina el resultado.
    """
    valores = _valores(datos)
    if valores.size < 2:
        raise ValueError("La región tiene menos de dos píxeles.")
    alto = float(np.percentile(valores, 95))
    bajo = float(np.percentile(valores, 5))
    if abs(alto + bajo) < 1e-12:
        return 0.0
    return 1.0 - (alto - bajo) / (alto + bajo)


def metricas_sin_referencia(img, lista_rois=None):
    """
    SNR, CNR y uniformidad (más contraste y ruido global).
      snr_roiN      = media / desvío de esa ROI.
      uniformidad_roiN = uniformidad integral de esa ROI.
      cnr_roiI_roiJ = |μI - μJ| / √((σI² + σJ²) / 2).
      uniformidad_entre_rois = 1 - (max - min) / (max + min) de las medias.
    """
    img = np.asarray(img, dtype=float)
    metricas = {
        "media": float(img.mean()),
        "contraste_rms": float(img.std()),
        "rango_p1_p99": float(np.percentile(img, 99) - np.percentile(img, 1)),
        "sigma_wavelet": descomposicion_wavelet(img, "db2", 1)["sigma_donoho"],
        "uniformidad_global": uniformidad(img),
        "fondo_borde_centro": indicador_fondo(img)["relacion_borde_centro"],
    }

    estadisticas = []
    for indice, datos in lista_rois or []:
        datos = _valores(datos)
        if datos.size < 2:
            continue
        media, desvio = float(datos.mean()), float(datos.std(ddof=1))
        estadisticas.append((indice, media, desvio))
        if desvio > 0:
            metricas[f"snr_roi{indice + 1}"] = media / desvio
        metricas[f"uniformidad_roi{indice + 1}"] = uniformidad(datos)

    for (i, media_i, desvio_i), (j, media_j, desvio_j) in combinations(estadisticas, 2):
        ruido = np.sqrt((desvio_i ** 2 + desvio_j ** 2) / 2)
        if ruido > 0:
            metricas[f"cnr_roi{i + 1}_roi{j + 1}"] = abs(media_i - media_j) / ruido

    if len(estadisticas) >= 2:
        medias = np.array([m for _, m, _ in estadisticas])
        suma = medias.max() + medias.min()
        if abs(suma) > 1e-12:
            metricas["uniformidad_entre_rois"] = 1.0 - (
                medias.max() - medias.min()
            ) / suma

    return metricas


def metricas_con_referencia(img, referencia, rango_datos=None):
    """
    MSE, RMSE, MAE, PSNR y SSIM contra una imagen de referencia.
      MSE/RMSE/MAE: error medio píxel a píxel (0 = idénticas).
      PSNR = 10·log10(rango² / MSE) en dB: crece al achicarse el error.
      SSIM: compara luminancia, contraste y estructura en ventanas
        locales; 1 = idénticas. Devuelve también el mapa por píxel.
    `rango_datos` es el rango dinámico usado por PSNR y SSIM.
    """
    img = np.asarray(img, dtype=float)
    referencia = np.asarray(referencia, dtype=float)
    if img.shape != referencia.shape:
        raise ValueError(
            f"La imagen {img.shape} y la referencia {referencia.shape} "
            "tienen tamaños distintos."
        )

    if rango_datos is None:
        rango_datos = float(
            max(img.max(), referencia.max()) - min(img.min(), referencia.min())
        )
    rango_datos = max(float(rango_datos), 1e-12)

    error = img - referencia
    mse = float(np.mean(error ** 2))
    psnr = 10 * np.log10(rango_datos ** 2 / mse) if mse > 0 else np.inf
    ssim, mapa = structural_similarity(
        referencia, img, data_range=rango_datos, full=True
    )

    correlacion = 0.0
    if img.std() > 0 and referencia.std() > 0:
        correlacion = float(np.corrcoef(img.ravel(), referencia.ravel())[0, 1])

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "psnr_db": float(psnr),
        "ssim": float(ssim),
        "correlacion": correlacion,
        "rango_datos": rango_datos,
    }, mapa


def evaluar(imagen, referencia=None, lista_rois=None, rango_datos=None):
    """
    Métricas de la Actividad 7: siempre las que no necesitan referencia
    y, si se pasa una, también las de comparación.
    Devuelve (metricas_sin_referencia, metricas_con_referencia, mapa_ssim).
    """
    img = a_escala_de_grises(imagen)
    sin_referencia = metricas_sin_referencia(img, lista_rois)

    if referencia is None:
        return sin_referencia, {}, None

    con_referencia, mapa = metricas_con_referencia(
        img, a_escala_de_grises(referencia), rango_datos
    )
    return sin_referencia, con_referencia, mapa


# ============================================================
# 4. VISOR Y ESTADO
# ============================================================

viewer = napari.Viewer()

registros_pendientes = []

informe = TextEdit(value="INFORME\nTodavía no se realizó ningún cálculo.")
informe.native.setReadOnly(True)
informe.native.setMinimumHeight(180)
estado_csv = Label(value="CSV\nNo hay resultados pendientes.")


def agregar_pendientes(registros):
    """Suma filas a las pendientes de guardar en el CSV."""
    registros_pendientes.extend(registros)
    estado_csv.value = (
        f"CSV\n{len(registros_pendientes)} fila(s) pendiente(s) de guardado."
    )


# ============================================================
# 5. CAPA DE ROIs (base de la clase, para Actividades 1 y 2)
#    Cada ROI puede llevar una etiqueta con el segmento que representa
#    (fondo, vaso, disco óptico...). Se muestra sobre la imagen y se
#    guarda en la columna "etiqueta_roi" del CSV.
# ============================================================

PALETA_ROIS = np.array([
    [1.00, 0.20, 0.20, 1.00],   # rojo
    [0.10, 0.80, 1.00, 1.00],   # celeste
    [0.20, 0.90, 0.35, 1.00],   # verde
    [1.00, 0.75, 0.10, 1.00],   # naranja
    [0.75, 0.35, 1.00, 1.00],   # violeta
    [1.00, 0.35, 0.75, 1.00],   # magenta
])


def obtener_capa_roi():
    if "ROI" not in viewer.layers:
        return None
    capa = viewer.layers["ROI"]
    if not isinstance(capa, Shapes):
        return None
    return capa


def etiqueta_roi(roi_layer, indice):
    """Etiqueta (segmento) de una ROI, o "" si no tiene."""
    if roi_layer is None or "etiqueta" not in roi_layer.features:
        return ""
    valor = roi_layer.features["etiqueta"].iloc[indice]
    return "" if valor is None or str(valor) == "nan" else str(valor)


def nombre_roi(roi_layer, indice):
    """'ROI 2' o 'ROI 2 (fondo)'."""
    etiqueta = etiqueta_roi(roi_layer, indice)
    return f"ROI {indice + 1}" + (f" ({etiqueta})" if etiqueta else "")


def actualizar_rois(event=None):
    """Colores por ROI y rótulo visible 'ROI n (etiqueta)'."""
    roi_layer = obtener_capa_roi()
    if roi_layer is None:
        return
    n = len(roi_layer.data)
    if n == 0:
        return
    colores_borde = np.array(
        [PALETA_ROIS[i % len(PALETA_ROIS)] for i in range(n)]
    )
    colores_cara = colores_borde.copy()
    colores_cara[:, 3] = 0.12
    roi_layer.edge_color = colores_borde
    roi_layer.face_color = colores_cara
    roi_layer.edge_width = 3

    features = roi_layer.features.copy()
    if "etiqueta" not in features:
        features["etiqueta"] = ""
    features["etiqueta"] = features["etiqueta"].fillna("").astype(str)
    features["rotulo"] = [
        f"ROI {i + 1}" + (f" ({e})" if e else "")
        for i, e in enumerate(features["etiqueta"])
    ]
    roi_layer.features = features


boton_crear_roi = PushButton(text="Crear capa de ROIs")


@boton_crear_roi.clicked.connect
def crear_capa_rois():
    roi_layer = obtener_capa_roi()

    if roi_layer is None:
        roi_layer = viewer.add_shapes(
            name="ROI",
            edge_width=3,
            features={"etiqueta": np.array([], dtype=object),
                      "rotulo": np.array([], dtype=object)},
            feature_defaults={"etiqueta": "", "rotulo": ""},
            text={"string": "{rotulo}", "color": "white", "size": 9,
                  "anchor": "upper_left"},
        )
        roi_layer.events.data.connect(actualizar_rois)
        mensaje = "Capa ROI creada.\nDibujá una o más regiones."
    else:
        mensaje = "La capa ROI ya existe.\nSe volvió a activar."

    viewer.layers.selection.active = roi_layer
    for widget in (widget_medicion, widget_gris, widget_diagnostico,
                   widget_ruido, widget_filtro, widget_deconvolucion,
                   widget_notch, widget_fondo, widget_realce,
                   widget_metricas):
        widget.roi.value = roi_layer
    roi_layer.mode = "add_rectangle"

    informe.value = f"INFORME\n{mensaje}"


etiqueta_nueva = LineEdit(value="fondo", label="Etiqueta (segmento)")
boton_etiquetar = PushButton(text="Asignar etiqueta a las ROIs seleccionadas")


@boton_etiquetar.clicked.connect
def etiquetar_rois():
    roi_layer = obtener_capa_roi()
    try:
        if roi_layer is None or len(roi_layer.data) == 0:
            raise ValueError("Primero creá la capa de ROIs y dibujá alguna ROI.")
        seleccionadas = sorted(roi_layer.selected_data)
        if not seleccionadas:
            raise ValueError("Seleccioná una o más ROIs con la herramienta Select.")

        actualizar_rois()   # asegura que existan las columnas
        features = roi_layer.features.copy()
        etiqueta = etiqueta_nueva.value.strip()
        for i in seleccionadas:
            features.loc[i, "etiqueta"] = etiqueta
        roi_layer.features = features
        actualizar_rois()

        informe.value = (
            "INFORME\nEtiqueta asignada:\n"
            + "\n".join(nombre_roi(roi_layer, i) for i in seleccionadas)
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo etiquetar:\n{error}"


# ============================================================
# 6. HERRAMIENTA DE MEDICIÓN (Actividades 1 y 2)
#    Media / varianza / desvío / MAD sobre la imagen completa,
#    la ROI seleccionada o todas las ROIs (una fila del CSV por región).
# ============================================================

@magicgui(
    call_button="Medir estadísticas",
    modo={
        "choices": ["Imagen completa", "ROI seleccionada", "Todas las ROIs"],
        "label": "Región a analizar",
    },
    excluir_marco={"label": "Quitar marco (imagen completa)"},
    registrar={"label": "Agregar al CSV"},
)
def widget_medicion(
    imagen: Image,
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        img = a_escala_de_grises(imagen.data)

        # (region, etiqueta, pixeles)
        if modo == "Imagen completa":
            datos = quitar_marco(img)[0] if excluir_marco else img
            region = "Imagen completa" + (" (sin marco)" if excluir_marco else "")
            regiones = [(region, "", datos)]
        else:
            if roi is None:
                raise ValueError("Creá primero una capa de ROIs.")
            if modo == "ROI seleccionada":
                indices = [obtener_indice_roi_seleccionada(roi)]
            else:
                if len(roi.data) == 0:
                    raise ValueError("Dibujá al menos una ROI.")
                indices = range(len(roi.data))
            origen = origen_capa(imagen)
            regiones = [
                (f"ROI {i + 1}", etiqueta_roi(roi, i),
                 obtener_datos_roi(img, roi, i, origen))
                for i in indices
            ]

        bloques = []
        registros = []
        for region, etiqueta, datos in regiones:
            est = estadisticas_robustas(datos)
            titulo_region = region + (f" ({etiqueta})" if etiqueta else "")
            bloques.append(
                f"{titulo_region}\n"
                f"  n = {est['n']}   media = {est['media']:.3f}\n"
                f"  varianza = {est['varianza']:.3f}   desvío = {est['desviacion']:.3f}\n"
                f"  mediana = {est['mediana']:.3f}   MAD = {est['mad']:.3f}"
                f"   σ_MAD = {est['sigma_mad']:.3f}"
            )
            registros.append(crear_registro(
                imagen=imagen.name,
                actividad="Medición",
                region=region,
                etiqueta_roi=etiqueta,
                metodo="Estadísticas de la región",
                n_px=est["n"],
                media=est["media"],
                desvio=est["desviacion"],
                mediana=est["mediana"],
                sigma_mad=est["sigma_mad"],
                resultados={"varianza": est["varianza"], "mad": est["mad"]},
            ))

        if registrar:
            agregar_pendientes(registros)

        informe.value = (
            "INFORME - MEDICIÓN\n"
            f"Capa: {imagen.name}\n\n"
            + "\n\n".join(bloques)
            + (f"\n\n{len(registros)} fila(s) agregada(s) al CSV pendiente."
               if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo calcular:\n{error}"


# ============================================================
# 6.1 CONVERSIÓN A ESCALA DE GRISES (paso previo a la Act. 1)
#     Las Retina*.png vienen como RGBA con R = G = B: se pasan a 2D.
#     Sobre la imagen completa o sobre el recorte de la ROI.
# ============================================================

def origen_capa(capa):
    """Desplazamiento (fila, columna) de una capa en el mundo."""
    return np.asarray(capa.translate, dtype=float)[-2:]


def agregar_o_actualizar_capa(nombre, datos, **kwargs):
    """Si la capa existe le cambia los datos; si no, la crea."""
    if nombre in viewer.layers:
        capa = viewer.layers[nombre]
        capa.data = datos
        for clave, valor in kwargs.items():
            setattr(capa, clave, valor)
        return capa
    return viewer.add_image(datos, name=nombre, **kwargs)


def region_en_gris(imagen, modo, roi, excluir_marco):
    """
    Devuelve (imagen_gris_2D, origen, descripcion) de la región pedida,
    con origen en coordenadas del mundo de napari.
    """
    gris = a_escala_de_grises(imagen.data)
    origen = origen_capa(imagen)

    if modo == "Imagen completa":
        region, desplazamiento = gris, (0, 0)
        descripcion = "Imagen completa"
    else:
        if roi is None:
            raise ValueError("Creá primero una capa de ROIs.")
        indice = obtener_indice_roi_seleccionada(roi)
        vertices = np.asarray(roi.data[indice]) - origen
        region, desplazamiento = recortar_roi(gris, vertices)
        descripcion = nombre_roi(roi, indice)

    if excluir_marco:
        region, (df, dc) = quitar_marco(region)
        desplazamiento = (desplazamiento[0] + df, desplazamiento[1] + dc)
        descripcion += " (sin marco)"

    return region, origen + np.asarray(desplazamiento, dtype=float), descripcion


def region_y_etiqueta(modo, roi):
    """('Imagen completa', '') o ('ROI 2', 'fondo'), para las columnas del CSV."""
    if modo == "Imagen completa":
        return "Imagen completa", ""
    indice = obtener_indice_roi_seleccionada(roi)
    return f"ROI {indice + 1}", etiqueta_roi(roi, indice)


@magicgui(
    call_button="Convertir a escala de grises",
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a convertir",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
)
def widget_gris(
    imagen: Image,
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = False,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        if modo == "Imagen completa":
            nombre = f"{imagen.name}_gris"
        else:
            nombre = f"{imagen.name}_{descripcion.split(' (')[0].replace(' ', '')}_gris"
        capa = agregar_o_actualizar_capa(
            nombre,
            region,
            colormap="gray",
            translate=tuple(origen),
        )
        viewer.layers.selection.active = capa

        informe.value = (
            "INFORME\n"
            f"Capa creada: {nombre}\n"
            f"Región: {descripcion}\n"
            f"Tamaño: {region.shape[0]} x {region.shape[1]} (2D, float)\n"
            f"Rango: {region.min():.1f} - {region.max():.1f}"
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo convertir:\n{error}"


# ============================================================
# 6.2 DIAGNÓSTICO DE LA IMAGEN (Actividad 1)
#     Histogramas, perfiles, estadísticos + MAD, variabilidad local,
#     diferencias, Fourier, autocorrelación y wavelet, más una
#     hipótesis sugerida de la degradación dominante.
# ============================================================

ventanas_figuras = []   # referencias para que Qt no cierre las ventanas


def obtener_capa_perfil():
    if "Perfil" not in viewer.layers:
        return None
    capa = viewer.layers["Perfil"]
    if not isinstance(capa, Shapes):
        return None
    return capa


boton_crear_perfil = PushButton(text="Crear capa de perfil (línea)")


@boton_crear_perfil.clicked.connect
def crear_capa_perfil():
    capa = obtener_capa_perfil()
    if capa is None:
        capa = viewer.add_shapes(
            name="Perfil", shape_type="line", edge_color="yellow", edge_width=2
        )
    viewer.layers.selection.active = capa
    capa.mode = "add_line"
    informe.value = (
        "INFORME\nDibujá una línea en la capa 'Perfil'.\n"
        "Se usa la última línea para el perfil de intensidad."
    )


def perfiles_intensidad(gris_completa, origen, region, origen_region):
    """
    Perfiles de intensidad. Si hay una línea en la capa 'Perfil' se usa;
    si no, se toman la fila y la columna centrales de la región.
    Devuelve una lista de (etiqueta, valores).
    """
    capa = obtener_capa_perfil()
    lineas = []
    if capa is not None:
        lineas = [
            np.asarray(d) for d, tipo in zip(capa.data, capa.shape_type)
            if tipo in ("line", "path") and len(d) >= 2
        ]

    if lineas:
        vertices = lineas[-1] - origen
        valores = profile_line(
            gris_completa, vertices[0], vertices[-1],
            linewidth=1, mode="constant", cval=np.nan,
        )
        return [("línea 'Perfil'", valores)]

    fila, columna = region.shape[0] // 2, region.shape[1] // 2
    f_abs = int(origen_region[0] + fila)
    c_abs = int(origen_region[1] + columna)
    return [
        (f"horizontal (fila {f_abs})", region[fila, :]),
        (f"vertical (columna {c_abs})", region[:, columna]),
    ]


def histogramas_regionales(gris_completa, origen):
    """Píxeles de cada ROI dibujada (para comparar con el global)."""
    roi_layer = obtener_capa_roi()
    if roi_layer is None:
        return []
    resultado = []
    for i in range(len(roi_layer.data)):
        try:
            datos = obtener_datos_roi(gris_completa, roi_layer, i, origen)
        except ValueError:
            continue
        if datos.size > 1:
            resultado.append((i, datos))
    return resultado


def figura_diagnostico(nombre, res, perfiles, regionales):
    """Arma la figura resumen de la Actividad 1 (4 x 4 paneles)."""
    fig = Figure(figsize=(17, 13), layout="constrained")
    ejes = fig.subplots(4, 4)
    img = res["imagen"]
    est = res["estadisticas"]

    # --- Fila 1: imagen, histogramas, perfiles
    ax = ejes[0, 0]
    ax.imshow(img, cmap="gray")
    ax.set_title("Región analizada (gris)")

    ax = ejes[0, 1]
    bins = np.linspace(img.min(), img.max() + 1e-9, 128)
    ax.hist(img.ravel(), bins=bins, density=True, color="0.4", label="global")
    for i, datos in regionales:
        ax.hist(
            datos, bins=bins, density=True, histtype="step", linewidth=1.5,
            color=PALETA_ROIS[i % len(PALETA_ROIS)],
            label=nombre_roi(obtener_capa_roi(), i),
        )
    ax.axvline(est["media"], color="k", linestyle="--", linewidth=1)
    ax.set_title("Histograma global y regional")
    ax.set_xlabel("intensidad")
    ax.legend(fontsize=7)

    ax = ejes[0, 2]
    for etiqueta, valores in perfiles:
        ax.plot(valores, linewidth=0.8, label=etiqueta)
    ax.set_title("Perfiles de intensidad")
    ax.set_xlabel("posición (px)")
    ax.legend(fontsize=7)

    ax = ejes[0, 3]
    ax.axis("off")
    ind = res["indicadores"]
    texto = (
        f"n = {est['n']}\n"
        f"media = {est['media']:.2f}\n"
        f"varianza = {est['varianza']:.2f}\n"
        f"desvío = {est['desviacion']:.2f}\n"
        f"mediana = {est['mediana']:.2f}\n"
        f"MAD = {est['mad']:.2f} (σ_MAD = {est['sigma_mad']:.2f})\n\n"
        f"σ diferencias (std) = {res['diferencias']['sigma_std']:.2f}\n"
        f"σ diferencias (MAD) = {res['diferencias']['sigma_mad']:.2f}\n"
        f"σ wavelet (Donoho) = {res['wavelet']['sigma_donoho']:.2f}\n"
        f"σ wavelet / rango = {ind['sigma_wavelet_sobre_rango']:.3f}\n"
        f"curtosis diferencias = {ind['curtosis_diferencias']:.1f}\n"
        f"impulsos = {100 * ind['fraccion_impulsos']:.2f} %\n"
        f"pico FFT = {ind['pico_fft_prominencia']:.2f} "
        f"(período {ind['pico_fft_periodo_px']:.1f} px)\n"
        f"fondo borde/centro = {ind['fondo_borde_centro']:.2f}\n"
        f"E altas / E medias = {ind['relacion_altas_medias']:.3f}"
    )
    ax.text(0, 1, texto, va="top", family="monospace", fontsize=9)
    ax.set_title(nombre)

    # --- Fila 2: variabilidad local, diferencias, fondo
    ax = ejes[1, 0]
    im = ax.imshow(res["variabilidad"], cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Variabilidad local (desvío)")

    ax = ejes[1, 1]
    im = ax.imshow(np.abs(res["diferencias"]["dx"]), cmap="magma",
                   vmax=np.percentile(np.abs(res["diferencias"]["dx"]), 99))
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("|Diferencia horizontal|")

    ax = ejes[1, 2]
    dx = res["diferencias"]["dx"].ravel()
    dy = res["diferencias"]["dy"].ravel()
    lim = np.ceil(np.percentile(np.abs(np.concatenate([dx, dy])), 99.5))
    bins_d = np.arange(-lim - 0.5, lim + 1.5, 1)   # centrados en enteros
    ax.hist(dx, bins=bins_d, histtype="step", label="dx", density=True)
    ax.hist(dy, bins=bins_d, histtype="step", label="dy", density=True)
    ax.set_yscale("log")
    ax.set_title("Histograma de diferencias")
    ax.legend(fontsize=7)

    ax = ejes[1, 3]
    im = ax.imshow(res["fondo"]["fondo"], cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Fondo (gaussiano de gran escala)")

    # --- Fila 3: Fourier y autocorrelación
    fourier = res["fourier"]
    ax = ejes[2, 0]
    ax.imshow(fourier["log_magnitud"], cmap="gray")
    fila, columna = fourier["pico_posicion"]
    ax.plot(columna, fila, "o", markerfacecolor="none", color="r", markersize=10)
    ax.set_title("log |FFT| (o = pico más prominente)")

    ax = ejes[2, 1]
    ax.semilogy(fourier["frecuencias_radiales"][1:], fourier["psd_radial"][1:])
    ax.set_xlabel("frecuencia (ciclos/px)")
    ax.set_title("Espectro de potencia radial")

    auto = res["autocorrelacion"]
    ax = ejes[2, 2]
    alto, ancho = auto["mapa"].shape
    lag = min(40, alto // 2 - 1, ancho // 2 - 1)
    ax.imshow(
        auto["mapa"][alto // 2 - lag:alto // 2 + lag + 1,
                     ancho // 2 - lag:ancho // 2 + lag + 1],
        cmap="coolwarm", vmin=-1, vmax=1, extent=(-lag, lag, lag, -lag),
    )
    ax.set_title("Autocorrelación (zoom)")

    ax = ejes[2, 3]
    ax.plot(auto["perfil_x"][:lag + 1], label="horizontal")
    ax.plot(auto["perfil_y"][:lag + 1], label="vertical")
    ax.axhline(0.5, color="0.6", linestyle=":")
    ax.set_xlabel("lag (px)")
    ax.set_title(f"Autocorrelación (lag1 = {auto['lag1_x']:.2f} / {auto['lag1_y']:.2f})")
    ax.legend(fontsize=7)

    # --- Fila 4: wavelet
    ondita = res["wavelet"]
    for ax, clave, titulo in [
        (ejes[3, 0], "horizontal_1", "Wavelet detalle H (nivel 1)"),
        (ejes[3, 1], "vertical_1", "Wavelet detalle V (nivel 1)"),
        (ejes[3, 2], "diagonal_1", "Wavelet detalle D (nivel 1)"),
    ]:
        coef = ondita[clave]
        lim = np.percentile(np.abs(coef), 99)
        ax.imshow(coef, cmap="gray", vmin=-lim, vmax=lim)
        ax.set_title(titulo)

    ax = ejes[3, 3]
    niveles = np.arange(1, len(ondita["energias_detalle"]) + 1)
    ax.bar(niveles, ondita["energias_detalle"], color="0.4")
    ax.set_yscale("log")
    ax.set_xticks(niveles)
    ax.set_xlabel("nivel (1 = más fino)")
    ax.set_title("Energía de detalle por nivel")

    for ax in [ejes[0, 0], ejes[1, 0], ejes[1, 1], ejes[1, 3],
               ejes[2, 0], ejes[3, 0], ejes[3, 1], ejes[3, 2]]:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


def mostrar_figura(fig, titulo):
    """
    Muestra una figura de matplotlib en una ventana Qt aparte.
    El lienzo va dentro de un QScrollArea con su tamaño real, así la
    ventana puede ser más chica que la pantalla sin cortar nada:
    si no entra, aparecen barras de desplazamiento.
    """
    ventana = QWidget()
    ventana.setWindowTitle(titulo)
    layout = QVBoxLayout(ventana)

    canvas = FigureCanvasQTAgg(fig)
    ancho_px = int(fig.get_figwidth() * fig.dpi)
    alto_px = int(fig.get_figheight() * fig.dpi)
    canvas.setMinimumSize(ancho_px, alto_px)   # conserva el tamaño real

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(canvas)

    layout.addWidget(NavigationToolbar2QT(canvas, ventana))
    layout.addWidget(scroll)

    ventana.resize(1200, 800)   # entra en pantallas de notebook
    ventana.show()
    ventanas_figuras.append(ventana)


@magicgui(
    call_button="Diagnosticar",
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a analizar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    ventana={"label": "Ventana variab. local", "min": 3, "max": 51, "step": 2},
    wavelet={"choices": ["haar", "db2", "db4", "sym4"], "label": "Wavelet"},
    niveles={"label": "Niveles wavelet", "min": 1, "max": 6},
    agregar_mapas={"label": "Agregar mapas como capas"},
    guardar_figura={"label": "Guardar figura en resultados/"},
    registrar={"label": "Agregar al CSV"},
)
def widget_diagnostico(
    imagen: Image,
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    ventana: int = 7,
    wavelet: str = "db2",
    niveles: int = 3,
    agregar_mapas: bool = True,
    guardar_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        res = diagnosticar(region, ventana=ventana, wavelet=wavelet, niveles=niveles)

        gris_completa = a_escala_de_grises(imagen.data)
        origen = origen_capa(imagen)
        perfiles = perfiles_intensidad(gris_completa, origen, region, origen_region)
        regionales = histogramas_regionales(gris_completa, origen)

        nombre = f"{imagen.name} - {descripcion}"
        fig = figura_diagnostico(nombre, res, perfiles, regionales)
        mostrar_figura(fig, f"Diagnóstico: {nombre}")

        ruta_figura = ""
        if guardar_figura:
            carpeta = Path("resultados")
            carpeta.mkdir(exist_ok=True)
            sufijo = descripcion.split(" (")[0].replace(" ", "")
            ruta_figura = carpeta / f"diagnostico_{imagen.name}_{sufijo}.png"
            fig.savefig(ruta_figura, dpi=110)

        if agregar_mapas:
            base = f"{imagen.name}_diag"
            traslado = tuple(origen_region)
            agregar_o_actualizar_capa(
                f"{base}_variabilidad", res["variabilidad"],
                colormap="magma", translate=traslado, visible=False,
            )
            agregar_o_actualizar_capa(
                f"{base}_fondo", res["fondo"]["fondo"],
                colormap="viridis", translate=traslado, visible=False,
            )
            agregar_o_actualizar_capa(
                f"{base}_fft_log", res["fourier"]["log_magnitud"],
                colormap="gray", visible=False,
            )
            agregar_o_actualizar_capa(
                f"{base}_autocorrelacion", res["autocorrelacion"]["mapa"],
                colormap="twilight_shifted", visible=False,
            )

        est = res["estadisticas"]
        ind = res["indicadores"]

        if registrar:
            region_fila, etiqueta = region_y_etiqueta(modo, roi)
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="1 - Diagnóstico",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo="Diagnóstico",
                parametros={
                    "tamano": f"{region.shape[0]}x{region.shape[1]}",
                    "sin_marco": excluir_marco,
                    "ventana": ventana,
                    "wavelet": wavelet,
                    "niveles": niveles,
                },
                n_px=est["n"],
                media=est["media"],
                desvio=est["desviacion"],
                mediana=est["mediana"],
                sigma_mad=est["sigma_mad"],
                resultados=ind,
                observaciones=str(ruta_figura),
            )])

        informe.value = (
            "INFORME - DIAGNÓSTICO\n"
            f"Capa: {imagen.name}\n"
            f"Región: {descripcion} ({region.shape[0]}x{region.shape[1]})\n"
            f"Media: {est['media']:.2f}   Varianza: {est['varianza']:.2f}\n"
            f"Desvío: {est['desviacion']:.2f}   Mediana: {est['mediana']:.2f}   "
            f"MAD: {est['mad']:.2f}\n\n"
            "INDICADORES\n"
            + "\n".join(
                f"  {clave} = {formatear_valor(valor)}" for clave, valor in ind.items()
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
            + ("\nFigura: " + str(ruta_figura) if ruta_figura else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo diagnosticar:\n{error}"


# ============================================================
# 6.3 ESTIMACIÓN DE RUIDO (Actividad 2)
#     Estrategia principal elegida por el usuario + las demás que se
#     puedan calcular con los datos disponibles, para comparar.
# ============================================================

# Última σ estimada: la Act. 3 la puede copiar como parámetro (a pedido).
ultimo_sigma_ruido = {}

REFERENCIA_LIMPIA = "Referencia sin ruido (σ = std(A-B))"
ADQUISICION_INDEPENDIENTE = "Adquisición independiente (σ = std(A-B)/√2)"


def figura_ruido(nombre, res):
    """Figura de la Actividad 2 (2 x 2 paneles)."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    metodo = res["metodo"]
    principal = res["principal"]
    sigma = res["sigma"]

    # --- Histograma de la muestra usada vs. N(centro, σ)
    ax = ejes[0, 0]
    muestra = principal["muestra"]
    centro = principal["centro"]
    ancho = max(5 * sigma, 3.0)
    if np.allclose(muestra, np.round(muestra)):
        paso = max(1, int(round(sigma / 3)))
        bins = np.arange(np.floor(centro - ancho) - 0.5, centro + ancho + 1, paso)
    else:
        bins = np.linspace(centro - ancho, centro + ancho, 61)
    ax.hist(muestra, bins=bins, density=True, color="0.6", label="datos")
    if sigma > 0:
        x = np.linspace(bins[0], bins[-1], 300)
        ax.plot(
            x, np.exp(-0.5 * ((x - centro) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
            "r", label=f"N({centro:.1f}, σ={sigma:.2f})",
        )
    if "media_fija" in principal:
        ax.axvline(principal["media_fija"], color="b", linestyle="--",
                   label=f"media fija = {principal['media_fija']:g}")
    ax.set_title(f"{metodo}: muestra usada vs. gaussiana")
    ax.legend(fontsize=7)

    # --- Comparación de estrategias
    ax = ejes[0, 1]
    nombres, valores, colores = [], [], []
    for nombre_m, r in res["resultados"].items():
        if "sigma" in r:
            nombres.append(nombre_m.replace(" entre ", "\nentre ").replace(" ", "\n", 1))
            valores.append(r["sigma"])
            colores.append("tab:red" if nombre_m == metodo else "0.5")
    ax.bar(range(len(valores)), valores, color=colores)
    ax.set_xticks(range(len(valores)))
    ax.set_xticklabels(nombres, fontsize=7)
    ax.set_ylabel("σ estimado")
    ax.set_title("Comparación de estrategias (rojo = elegida)")

    # --- Media vs. varianza de las ROIs (dependencia con la señal)
    ax = ejes[1, 0]
    multiples = res["resultados"]["Múltiples ROIs"]
    if "rois" in multiples:
        for indice, _, media, varianza, _ in multiples["rois"]:
            ax.plot(media, varianza, "o",
                    color=PALETA_ROIS[indice % len(PALETA_ROIS)])
            ax.annotate(nombre_roi(obtener_capa_roi(), indice),
                        (media, varianza), fontsize=7)
        aj = multiples["ajuste"]
        if aj is not None:
            medias = np.array([f[2] for f in multiples["rois"]])
            x = np.linspace(medias.min(), medias.max(), 50)
            ax.plot(x, aj["a"] + aj["b"] * x, "k--",
                    label=f"var = {aj['a']:.1f} + {aj['b']:.3f}·μ (R²={aj['r2']:.2f})")
            ax.legend(fontsize=7)
        ax.set_xlabel("media de la ROI")
        ax.set_ylabel("varianza de la ROI")
    else:
        ax.text(0.5, 0.5, "Dibujá 2 o más ROIs homogéneas\na distintas intensidades",
                ha="center", va="center")
    ax.set_title("Varianza vs. media por ROI")

    # --- Mapa del residuo de ruido
    ax = ejes[1, 1]
    if metodo == "Diferencias entre adquisiciones":
        mapa, titulo_mapa = principal["mapa"], "Diferencia entre adquisiciones (A - B)"
    elif "mapa" in res["resultados"]["Coeficientes wavelet"]:
        mapa = res["resultados"]["Coeficientes wavelet"]["mapa"]
        titulo_mapa = "Coeficientes wavelet HH1 (nivel 1)"
    else:
        mapa, titulo_mapa = None, ""
    if mapa is not None:
        lim = max(np.percentile(np.abs(mapa - np.median(mapa)), 99), 1e-9)
        im = ax.imshow(mapa, cmap="gray", vmin=np.median(mapa) - lim,
                       vmax=np.median(mapa) + lim)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(titulo_mapa)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(nombre)
    return fig


def clave_csv(texto):
    """'Dark-field / flat-field' -> 'dark_field_flat_field' (para claves y capas)."""
    reemplazos = str.maketrans("áéíóúñ /-", "aeioun___")
    clave = texto.lower().translate(reemplazos)
    while "__" in clave:
        clave = clave.replace("__", "_")
    return clave.strip("_")


@magicgui(
    call_button="Estimar ruido",
    metodo={"choices": METODOS_RUIDO, "label": "Estrategia"},
    modo={
        "choices": ["ROI seleccionada", "Imagen completa"],
        "label": "Región (dif. / wavelet)",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    usar_media_fija={"label": "Usar media fija (no la observada)"},
    media_fija={"label": "Media fija", "min": -1e6, "max": 1e6, "step": 1.0},
    segunda={"label": "Otra adquisición"},
    tipo_segunda={
        "choices": [REFERENCIA_LIMPIA, ADQUISICION_INDEPENDIENTE],
        "label": "La otra imagen es",
    },
    wavelet={"choices": ["haar", "db2", "db4", "sym4"], "label": "Wavelet"},
    ver_figura={"label": "Mostrar figura"},
    guardar_figura={"label": "Guardar figura en resultados/"},
    registrar={"label": "Agregar al CSV"},
)
def widget_ruido(
    imagen: Image,
    metodo: str = "ROI homogénea",
    modo: str = "ROI seleccionada",
    roi: Shapes = None,
    excluir_marco: bool = True,
    usar_media_fija: bool = False,
    media_fija: float = 0.0,
    segunda: Image = None,
    tipo_segunda: str = REFERENCIA_LIMPIA,
    wavelet: str = "db2",
    ver_figura: bool = True,
    guardar_figura: bool = False,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        gris = a_escala_de_grises(imagen.data)
        origen = origen_capa(imagen)
        notas = []
        motivos = {}

        # ROI seleccionada: píxeles del polígono y su rectángulo
        datos_roi = recorte_roi = None
        region_roi, etiqueta = "", ""
        try:
            indice = obtener_indice_roi_seleccionada(roi)
            datos_roi = obtener_datos_roi(gris, roi, indice, origen)
            recorte_roi, _ = recortar_roi(gris, np.asarray(roi.data[indice]) - origen)
            region_roi, etiqueta = f"ROI {indice + 1}", etiqueta_roi(roi, indice)
        except ValueError as error:
            motivos["ROI"] = str(error)

        # Región rectangular para diferencias locales / wavelet / adquisiciones
        modo_efectivo = modo
        if modo == "ROI seleccionada" and datos_roi is None:
            modo_efectivo = "Imagen completa"
            notas.append(
                "No hay ROI seleccionada: diferencias y wavelet se calcularon "
                "sobre la imagen completa."
            )
        region, origen_region, descripcion = region_en_gris(
            imagen, modo_efectivo, roi, excluir_marco
        )

        # Misma región en la otra adquisición
        region_b = None
        if segunda is None:
            motivos["segunda"] = "Elegí otra adquisición en 'Otra adquisición'."
        elif segunda is imagen:
            motivos["segunda"] = "La otra adquisición es la misma capa."
        else:
            gris_b = a_escala_de_grises(segunda.data)
            if gris_b.shape != gris.shape:
                motivos["segunda"] = (
                    f"'{segunda.name}' tiene otro tamaño ({gris_b.shape})."
                )
            else:
                f0, c0 = np.round(origen_region - origen).astype(int)
                alto, ancho = region.shape
                region_b = gris_b[f0:f0 + alto, c0:c0 + ancho]

        referencia_limpia = tipo_segunda == REFERENCIA_LIMPIA
        res = estimar_ruido(
            region,
            metodo=metodo,
            datos_roi=datos_roi,
            recorte_roi=recorte_roi,
            lista_rois=histogramas_regionales(gris, origen),
            segunda=region_b,
            referencia_limpia=referencia_limpia,
            media_fija=media_fija if usar_media_fija else None,
            wavelet=wavelet,
            motivos=motivos,
        )
        principal = res["principal"]
        sigma = res["sigma"]

        imagen_util = quitar_marco(gris)[0] if excluir_marco else gris
        rango = np.percentile(imagen_util, 99) - np.percentile(imagen_util, 1)
        roi_layer = obtener_capa_roi()

        # --- Región y etiqueta de la fila principal
        if metodo in ("ROI homogénea", "MAD"):
            region_fila = region_roi
            etiqueta_fila = etiqueta
            donde = f"{nombre_roi(roi_layer, indice)} ({principal['n']} px)"
        elif metodo == "Múltiples ROIs":
            indices = [f[0] for f in principal["rois"]]
            region_fila = ", ".join(f"ROI {i + 1}" for i in indices)
            etiquetas = [etiqueta_roi(roi_layer, i) for i in indices]
            etiqueta_fila = ", ".join(sorted({e for e in etiquetas if e}))
            donde = ", ".join(nombre_roi(roi_layer, i) for i in indices)
        else:
            region_fila, etiqueta_fila = region_y_etiqueta(modo_efectivo, roi)
            donde = f"{descripcion} ({region.shape[0]}x{region.shape[1]})"
        if metodo == "Diferencias entre adquisiciones":
            donde = f"{imagen.name} - {segunda.name}, {donde}"

        # --- Valores propios de la estrategia
        valores = {"sigma_sobre_rango": sigma / max(rango, 1e-12)}
        detalle = []
        if metodo == "ROI homogénea":
            valores["snr_media_sobre_sigma"] = principal["snr"]
            detalle.append(
                f"Media observada = {principal['media']:.2f}   "
                f"SNR = media/σ = {principal['snr']:.1f}"
            )
        elif metodo == "MAD":
            valores["mad"] = principal["mad"]
            detalle.append(
                f"Mediana = {principal['mediana']:.2f}   MAD = {principal['mad']:.2f}"
            )
        elif metodo == "Múltiples ROIs":
            valores["cv_sigma_entre_rois"] = principal["cv_sigma"]
            for i, n, media, _, desvio in principal["rois"]:
                detalle.append(
                    f"{nombre_roi(roi_layer, i)}: n={n}  media={media:.2f}  σ={desvio:.2f}"
                )
            detalle.append(f"CV de σ entre ROIs = {principal['cv_sigma']:.3f}")
            aj = principal["ajuste"]
            if aj is not None:
                valores.update(var_vs_media_a=aj["a"], var_vs_media_b=aj["b"],
                               var_vs_media_r2=aj["r2"])
                detalle.append(
                    f"Ajuste varianza = a + b·media: a = {aj['a']:.3f}, "
                    f"b = {aj['b']:.4f}, R² = {aj['r2']:.3f}"
                )
        elif metodo == "Diferencias entre adquisiciones":
            valores.update(sigma_std=principal["sigma_std"],
                           offset_medio=principal["offset"],
                           correlacion=principal["correlacion"])
            detalle.append(
                f"σ (MAD) = {sigma:.2f}   σ (std) = {principal['sigma_std']:.2f}   "
                f"offset medio A-B = {principal['offset']:+.2f}   "
                f"correlación = {principal['correlacion']:.3f}"
            )
        elif metodo == "Diferencias locales":
            valores.update(sigma_std=principal["sigma_std"],
                           curtosis=principal["curtosis"])
            detalle.append(
                f"σ (MAD) = {sigma:.2f}   σ (std) = {principal['sigma_std']:.2f}   "
                f"curtosis = {principal['curtosis']:.2f}"
            )

        if "media_fija" in principal:
            valores.update(media_fija=principal["media_fija"],
                           delta=principal["delta"],
                           sigma_alrededor_media_fija=principal["sigma_media_fija"])
            nivel = "media" if metodo == "ROI homogénea" else "mediana"
            detalle.append(
                f"Media fija μ0 = {principal['media_fija']:g}:   "
                f"Δ = {nivel} - μ0 = {principal['delta']:+.2f}   "
                f"dispersión alrededor de μ0 = {principal['sigma_media_fija']:.2f}"
            )

        valores.update(res["complementarios"])
        for nombre_m, r in res["resultados"].items():
            if nombre_m != metodo and "sigma" in r:
                valores[f"sigma_por_{clave_csv(nombre_m)}"] = r["sigma"]

        ultimo_sigma_ruido.update(
            valor=sigma, imagen=imagen.name, metodo=metodo, region=donde
        )

        # --- Figura
        ruta_figura = ""
        if ver_figura or guardar_figura:
            nombre_fig = f"Ruido: {imagen.name} - {metodo}"
            fig = figura_ruido(nombre_fig, res)
            if ver_figura:
                mostrar_figura(fig, nombre_fig)
            if guardar_figura:
                carpeta = Path("resultados")
                carpeta.mkdir(exist_ok=True)
                ruta_figura = carpeta / f"ruido_{imagen.name}_{clave_csv(metodo)}.png"
                fig.savefig(ruta_figura, dpi=110)

        # --- Filas del CSV
        if registrar:
            parametros = {"sin_marco": excluir_marco}
            if usar_media_fija and metodo in ("ROI homogénea", "MAD"):
                parametros["media_fija"] = media_fija
            if metodo == "Coeficientes wavelet":
                parametros["wavelet"] = wavelet
            if metodo == "Diferencias entre adquisiciones":
                parametros["otra_adquisicion"] = segunda.name
                parametros["tipo"] = "referencia" if referencia_limpia else "independiente"

            estadisticas = {}
            if metodo in ("ROI homogénea", "MAD"):
                est = estadisticas_robustas(datos_roi)
                estadisticas = dict(
                    n_px=est["n"], media=est["media"], desvio=est["desviacion"],
                    mediana=est["mediana"], sigma_mad=est["sigma_mad"],
                )

            filas = []
            if metodo == "Múltiples ROIs":
                # una fila por ROI, para comparar niveles e intensidades
                for i, n, media, _, desvio in principal["rois"]:
                    est = estadisticas_robustas(
                        obtener_datos_roi(gris, roi_layer, i, origen)
                    )
                    filas.append(crear_registro(
                        imagen=imagen.name,
                        actividad="2 - Estimación de ruido",
                        region=f"ROI {i + 1}",
                        etiqueta_roi=etiqueta_roi(roi_layer, i),
                        metodo="Múltiples ROIs (cada ROI)",
                        n_px=n, media=media, desvio=desvio,
                        mediana=est["mediana"], sigma_mad=est["sigma_mad"],
                        estimacion_ruido=desvio,
                    ))
            filas.append(crear_registro(
                imagen=imagen.name,
                actividad="2 - Estimación de ruido",
                region=region_fila,
                etiqueta_roi=etiqueta_fila,
                metodo=metodo,
                parametros=parametros,
                estimacion_ruido=sigma,
                resultados=valores,
                observaciones=str(ruta_figura),
                **estadisticas,
            ))
            agregar_pendientes(filas)

        comparacion = []
        for nombre_m, r in res["resultados"].items():
            marca = "→" if nombre_m == metodo else " "
            valor = f"{r['sigma']:.3f}" if "sigma" in r else f"no aplica ({r['error']})"
            comparacion.append(f"{marca} {nombre_m}: {valor}")

        complementarios = [
            f"  {clave} = {formatear_valor(valor)}"
            for clave, valor in res["complementarios"].items()
        ]

        informe.value = (
            "INFORME - ESTIMACIÓN DE RUIDO (Act. 2)\n"
            f"Capa: {imagen.name}\n"
            f"Estrategia: {metodo}\n"
            f"Región: {donde}\n"
            f"σ = {sigma:.3f}   (σ / rango p1-p99 = {100 * valores['sigma_sobre_rango']:.2f}%)\n"
            + "".join(f"{d}\n" for d in detalle)
            + "\nCOMPARACIÓN DE ESTRATEGIAS (σ)\n"
            + "\n".join(comparacion)
            + "\n\nOTROS VALORES\n"
            + "\n".join(complementarios)
            + ("\n\nNOTAS\n" + "\n".join(f"- {n}" for n in notas) if notas else "")
            + (f"\n\n{len(filas)} fila(s) agregada(s) al CSV pendiente."
               if registrar else "")
            + ("\nFigura: " + str(ruta_figura) if ruta_figura else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo estimar el ruido:\n{error}"


# ============================================================
# 6.4 REDUCCIÓN DE RUIDO (Actividad 3)
#     Se elige el filtro y sus parámetros; el resultado es una capa
#     nueva (entrada de los pasos siguientes). El panel muestra solo
#     los parámetros del método elegido.
# ============================================================

# Parámetros que usa cada método (el resto se oculta en el panel)
PARAMETROS_FILTRO = {
    "Gaussiano": ["sigma_espacial"],
    "Bilateral": ["sigma_color", "sigma_espacial"],
    "Difusión anisotrópica": ["iteraciones", "kappa", "gamma", "funcion_conduccion"],
    "Non-Local Means": ["h", "sigma_ruido", "tamano_parche", "distancia_busqueda"],
    "Variación Total": ["peso_tv"],
    "Wavelet thresholding": [
        "sigma_ruido", "wavelet", "umbral", "modo_umbral", "niveles_wavelet",
    ],
    "Wiener local": ["ventana", "sigma_ruido"],
    "BM3D": ["sigma_ruido", "etapas_bm3d"],
}
TODOS_PARAMETROS_FILTRO = sorted(
    {nombre for nombres in PARAMETROS_FILTRO.values() for nombre in nombres}
)


def escala_intensidad(datos):
    """Máximo del tipo de dato (255 en uint8): pasa intensidades a [0, 1]."""
    datos = np.asarray(datos)
    if np.issubdtype(datos.dtype, np.integer):
        return float(np.iinfo(datos.dtype).max)
    return 1.0 if np.nanmax(datos) <= 1 else float(np.nanmax(datos))


def figura_filtro(nombre, original, filtrada):
    """Original, filtrada, residuo (original - filtrada) y perfil central."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    vmin, vmax = np.percentile(original, [0.5, 99.5])

    ax = ejes[0, 0]
    ax.imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Original")

    ax = ejes[0, 1]
    ax.imshow(filtrada, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Filtrada (misma escala de grises)")

    ax = ejes[1, 0]
    residuo = original - filtrada
    lim = max(np.percentile(np.abs(residuo), 99), 1e-9)
    im = ax.imshow(residuo, cmap="gray", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Residuo = original - filtrada (lo que se quitó)")

    for ax in (ejes[0, 0], ejes[0, 1], ejes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])

    ax = ejes[1, 1]
    fila = original.shape[0] // 2
    ax.plot(original[fila], color="0.6", linewidth=0.8, label="original")
    ax.plot(filtrada[fila], color="tab:red", linewidth=1.0, label="filtrada")
    ax.set_xlabel("columna (px)")
    ax.set_title(f"Perfil horizontal (fila central {fila})")
    ax.legend(fontsize=7)

    fig.suptitle(nombre)
    return fig


boton_usar_sigma = PushButton(text="Copiar σ de la última estimación (Act. 2)")


@boton_usar_sigma.clicked.connect
def copiar_ultimo_sigma():
    if not ultimo_sigma_ruido:
        informe.value = "INFORME\nTodavía no se estimó el ruido (Act. 2)."
        return
    widget_filtro.sigma_ruido.value = round(float(ultimo_sigma_ruido["valor"]), 3)
    informe.value = (
        "INFORME\nσ copiado al parámetro 'σ del ruido':\n"
        f"σ = {ultimo_sigma_ruido['valor']:.3f} ({ultimo_sigma_ruido['metodo']}, "
        f"{ultimo_sigma_ruido['imagen']}, {ultimo_sigma_ruido['region']})"
    )


@magicgui(
    call_button="Aplicar filtro",
    metodo={"choices": METODOS_FILTRO, "label": "Filtro"},
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a filtrar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    sigma_espacial={"label": "σ espacial (px)", "min": 0.1, "max": 20.0, "step": 0.1},
    sigma_color={"label": "σ color (intensidad)", "min": 0.1, "max": 1000.0, "step": 0.5},
    iteraciones={"label": "Iteraciones", "min": 1, "max": 500},
    kappa={"label": "κ (intensidad)", "min": 0.1, "max": 1000.0, "step": 0.5},
    gamma={"label": "γ (paso, ≤ 0.25)", "min": 0.01, "max": 0.25, "step": 0.01},
    funcion_conduccion={
        "choices": ["exponencial", "cuadrática"],
        "label": "Función de conducción",
    },
    h={"label": "h (intensidad)", "min": 0.1, "max": 1000.0, "step": 0.5},
    sigma_ruido={
        "label": "σ del ruido (0 = auto)", "min": 0.0, "max": 1000.0, "step": 0.1,
    },
    tamano_parche={"label": "Tamaño de parche (px)", "min": 3, "max": 21},
    distancia_busqueda={"label": "Distancia de búsqueda (px)", "min": 1, "max": 30},
    peso_tv={"label": "Peso TV", "min": 0.001, "max": 5.0, "step": 0.005},
    wavelet={"choices": ["haar", "db2", "db4", "sym4", "sym8", "coif1"], "label": "Wavelet"},
    umbral={"choices": ["BayesShrink", "VisuShrink"], "label": "Umbral"},
    modo_umbral={"choices": ["soft", "hard"], "label": "Umbralado"},
    niveles_wavelet={"label": "Niveles (0 = auto)", "min": 0, "max": 8},
    ventana={"label": "Ventana (px)", "min": 3, "max": 31, "step": 2},
    etapas_bm3d={
        "choices": ["Completo (2 etapas)", "Solo umbral duro (1 etapa)"],
        "label": "Etapas BM3D",
    },
    agregar_residuo={"label": "Agregar capa de residuo"},
    ver_figura={"label": "Mostrar figura comparativa"},
    registrar={"label": "Agregar al CSV"},
)
def widget_filtro(
    imagen: Image,
    metodo: str = "Gaussiano",
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    sigma_espacial: float = 1.0,
    sigma_color: float = 20.0,
    iteraciones: int = 15,
    kappa: float = 20.0,
    gamma: float = 0.2,
    funcion_conduccion: str = "exponencial",
    h: float = 10.0,
    sigma_ruido: float = 0.0,
    tamano_parche: int = 5,
    distancia_busqueda: int = 6,
    peso_tv: float = 0.05,
    wavelet: str = "db2",
    umbral: str = "BayesShrink",
    modo_umbral: str = "soft",
    niveles_wavelet: int = 0,
    ventana: int = 5,
    etapas_bm3d: str = "Completo (2 etapas)",
    agregar_residuo: bool = False,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        escala = escala_intensidad(imagen.data)

        todos = dict(
            sigma_espacial=sigma_espacial, sigma_color=sigma_color,
            iteraciones=iteraciones, kappa=kappa, gamma=gamma,
            funcion_conduccion=funcion_conduccion, h=h, sigma_ruido=sigma_ruido,
            tamano_parche=tamano_parche, distancia_busqueda=distancia_busqueda,
            peso_tv=peso_tv, wavelet=wavelet, umbral=umbral,
            modo_umbral=modo_umbral, niveles_wavelet=niveles_wavelet,
            ventana=ventana, etapas_bm3d=etapas_bm3d,
        )
        parametros = {k: todos[k] for k in PARAMETROS_FILTRO[metodo]}

        inicio = datetime.now()
        filtrada = reducir_ruido(region, metodo, escala=escala, **todos)
        segundos = (datetime.now() - inicio).total_seconds()

        # --- Capas nuevas (en la posición de la región original)
        nombre_capa = f"{imagen.name}_{clave_csv(metodo)}"
        capa = agregar_o_actualizar_capa(
            nombre_capa, filtrada, colormap="gray",
            translate=tuple(origen_region),
            contrast_limits=tuple(imagen.contrast_limits),
        )
        residuo = region - filtrada
        if agregar_residuo:
            lim = max(float(np.percentile(np.abs(residuo), 99)), 1e-9)
            agregar_o_actualizar_capa(
                f"{nombre_capa}_residuo", residuo, colormap="gray",
                translate=tuple(origen_region), contrast_limits=(-lim, lim),
                visible=False,
            )
        viewer.layers.selection.active = capa

        # --- Valores antes / después (sin interpretar)
        antes = {"sigma_wavelet": descomposicion_wavelet(region, "db2", 1)["sigma_donoho"]}
        despues = {"sigma_wavelet": descomposicion_wavelet(filtrada, "db2", 1)["sigma_donoho"]}

        # σ en la ROI seleccionada (si hay una y cae dentro de la región)
        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        roi_medida = ""
        try:
            indice = obtener_indice_roi_seleccionada(roi)
            gris = a_escala_de_grises(imagen.data)
            origen = origen_capa(imagen)
            filtrada_completa = gris.copy()
            f0, c0 = np.round(origen_region - origen).astype(int)
            filtrada_completa[f0:f0 + region.shape[0], c0:c0 + region.shape[1]] = filtrada
            en_region = np.zeros(gris.shape, dtype=bool)
            en_region[f0:f0 + region.shape[0], c0:c0 + region.shape[1]] = True
            mascara = obtener_datos_roi(en_region.astype(float), roi, indice, origen)
            if mascara.size >= 2 and mascara.all():
                datos_antes = obtener_datos_roi(gris, roi, indice, origen)
                datos_despues = obtener_datos_roi(filtrada_completa, roi, indice, origen)
                antes.update(media_roi=datos_antes.mean(), desvio_roi=datos_antes.std(ddof=1))
                despues.update(media_roi=datos_despues.mean(), desvio_roi=datos_despues.std(ddof=1))
                roi_medida = nombre_roi(roi, indice)
        except ValueError:
            pass

        resultados = {
            "tiempo_s": segundos,
            "residuo_media": float(residuo.mean()),
            "residuo_desvio": float(residuo.std()),
            "capa_resultado": nombre_capa,
        }
        if roi_medida:
            resultados["roi_medida"] = roi_medida

        ruta_figura = ""
        if ver_figura:
            nombre_fig = f"Filtro: {imagen.name} - {metodo}"
            fig = figura_filtro(
                f"{nombre_fig} ({formatear_resultados(parametros)})", region, filtrada
            )
            mostrar_figura(fig, nombre_fig)

        if registrar:
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="3 - Reducción de ruido",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=metodo,
                parametros=parametros,
                estimacion_ruido=sigma_ruido if "sigma_ruido" in parametros and sigma_ruido > 0 else "",
                metrica_antes=antes,
                metrica_despues=despues,
                resultados=resultados,
            )])

        lineas_medidas = [
            f"  {clave}: {formatear_valor(antes[clave])} → {formatear_valor(despues[clave])}"
            for clave in antes
        ]
        informe.value = (
            "INFORME - REDUCCIÓN DE RUIDO (Act. 3)\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Filtro: {metodo}\n"
            "Parámetros: " + formatear_resultados(parametros) + "\n"
            f"Tiempo: {segundos:.2f} s\n"
            f"Capa creada: {nombre_capa}\n\n"
            "ANTES → DESPUÉS\n"
            + "\n".join(lineas_medidas)
            + (f"\n  (ROI medida: {roi_medida})" if roi_medida else "")
            + f"\n\nResiduo (original - filtrada): media = {residuo.mean():.3f}, "
            f"desvío = {residuo.std():.3f}"
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo filtrar:\n{error}"


def actualizar_parametros_filtro(event=None):
    """Muestra solo los parámetros del método elegido."""
    visibles = PARAMETROS_FILTRO[widget_filtro.metodo.value]
    for nombre in TODOS_PARAMETROS_FILTRO:
        getattr(widget_filtro, nombre).visible = nombre in visibles


widget_filtro.metodo.changed.connect(actualizar_parametros_filtro)


# ============================================================
# 6.5 RESTAURACIÓN Y ARTEFACTOS (Actividad 4)
#     a) Deconvolución (Wiener / Richardson-Lucy) con PSF sintética
#        o provista en una capa.
#     b) Filtro notch: detección de picos en Fourier, diseño del notch
#        y valores para controlar qué se removió.
# ============================================================

PARAMETROS_PSF = {
    "Gaussiana": ["sigma_psf"],
    "Disco (desenfoque)": ["radio_psf"],
    "Movimiento lineal": ["longitud_psf", "angulo_psf"],
    "Capa de napari": ["capa_psf"],
}
PARAMETROS_DECONVOLUCION = {
    "Wiener": ["balance"],
    "Wiener no supervisado": [],
    "Richardson-Lucy": ["iteraciones", "tolerancia"],
}
TODOS_PARAMETROS_DECONV = sorted(
    {n for v in PARAMETROS_PSF.values() for n in v}
    | {n for v in PARAMETROS_DECONVOLUCION.values() for n in v}
)


def medidas_restauracion(img):
    """Valores para comparar antes y después (nitidez, ruido, contraste)."""
    gy, gx = np.gradient(np.asarray(img, dtype=float))
    return {
        "energia_gradiente": float(np.mean(gx ** 2 + gy ** 2)),
        "sigma_wavelet": descomposicion_wavelet(img, "db2", 1)["sigma_donoho"],
        "desvio": float(np.std(img)),
    }


def figura_deconvolucion(nombre, original, restaurada, psf, historial):
    """Original, restaurada, PSF y (RL) curvas de convergencia."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    vmin, vmax = np.percentile(original, [0.5, 99.5])

    ax = ejes[0, 0]
    ax.imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Original")

    ax = ejes[0, 1]
    ax.imshow(restaurada, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Restaurada (misma escala de grises)")

    ax = ejes[1, 0]
    im = ax.imshow(psf, cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"PSF {psf.shape[0]}x{psf.shape[1]} (suma = {psf.sum():.3f})")

    for ax in (ejes[0, 0], ejes[0, 1], ejes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])

    ax = ejes[1, 1]
    if historial:
        iteraciones = [h["iteracion"] for h in historial]
        ax.plot(iteraciones, [h["residuo_rms"] for h in historial],
                color="tab:blue", label="residuo RMS")
        ax.set_xlabel("iteración")
        ax.set_ylabel("residuo RMS", color="tab:blue")
        eje2 = ax.twinx()
        eje2.semilogy(iteraciones, [h["cambio"] for h in historial],
                      color="tab:red", label="cambio relativo")
        eje2.set_ylabel("cambio relativo", color="tab:red")
        ax.set_title("Convergencia de Richardson-Lucy")
    else:
        fila = original.shape[0] // 2
        ax.plot(original[fila], color="0.6", linewidth=0.8, label="original")
        ax.plot(restaurada[fila], color="tab:red", linewidth=1.0, label="restaurada")
        ax.set_xlabel("columna (px)")
        ax.set_title(f"Perfil horizontal (fila central {fila})")
        ax.legend(fontsize=7)

    fig.suptitle(nombre)
    return fig


boton_nsr = PushButton(text="Copiar NSR (σ²/Var) al balance")


@boton_nsr.clicked.connect
def copiar_nsr():
    try:
        if not ultimo_sigma_ruido:
            raise ValueError("Todavía no se estimó el ruido (Act. 2).")
        imagen = widget_deconvolucion.imagen.value
        if imagen is None:
            raise ValueError("Elegí la capa en el bloque de deconvolución.")
        gris = a_escala_de_grises(imagen.data)
        nsr = ultimo_sigma_ruido["valor"] ** 2 / max(float(gris.var()), 1e-12)
        widget_deconvolucion.balance.value = nsr
        informe.value = (
            "INFORME\nBalance = σ²/Var(imagen) = "
            f"{ultimo_sigma_ruido['valor']:.3f}² / {gris.var():.1f} = {nsr:.6f}\n"
            f"(σ de {ultimo_sigma_ruido['metodo']} sobre {ultimo_sigma_ruido['imagen']})"
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo calcular el NSR:\n{error}"


@magicgui(
    call_button="Aplicar deconvolución",
    metodo={"choices": METODOS_DECONVOLUCION, "label": "Método"},
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a restaurar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    tipo_psf={"choices": TIPOS_PSF, "label": "PSF"},
    capa_psf={"label": "Capa con la PSF"},
    sigma_psf={"label": "σ de la PSF (px)", "min": 0.1, "max": 50.0, "step": 0.1},
    radio_psf={"label": "Radio del disco (px)", "min": 0.5, "max": 50.0, "step": 0.5},
    longitud_psf={"label": "Longitud (px)", "min": 2.0, "max": 100.0, "step": 1.0},
    angulo_psf={"label": "Ángulo (grados)", "min": -180.0, "max": 180.0, "step": 5.0},
    balance={
        "label": "Balance (regularización)",
        "min": 1e-6, "max": 10.0, "step": 0.001,
    },
    iteraciones={"label": "Iteraciones máx.", "min": 1, "max": 500},
    tolerancia={
        "label": "Parar si cambio <", "min": 0.0, "max": 1.0, "step": 0.0005,
    },
    ver_figura={"label": "Mostrar figura comparativa"},
    registrar={"label": "Agregar al CSV"},
)
def widget_deconvolucion(
    imagen: Image,
    metodo: str = "Wiener",
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    tipo_psf: str = "Gaussiana",
    capa_psf: Image = None,
    sigma_psf: float = 2.0,
    radio_psf: float = 3.0,
    longitud_psf: float = 9.0,
    angulo_psf: float = 0.0,
    balance: float = 0.01,
    iteraciones: int = 20,
    tolerancia: float = 0.0,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        escala = escala_intensidad(imagen.data)

        psf = construir_psf(
            tipo_psf, sigma_psf, radio_psf, longitud_psf, angulo_psf,
            capa_psf.data if capa_psf is not None else None,
        )

        inicio = datetime.now()
        restaurada, info = restaurar(
            region, metodo=metodo, psf=psf, balance=balance,
            iteraciones=iteraciones, tolerancia=tolerancia, escala=escala,
        )
        segundos = (datetime.now() - inicio).total_seconds()

        nombre_capa = f"{imagen.name}_{clave_csv(metodo)}"
        capa = agregar_o_actualizar_capa(
            nombre_capa, restaurada, colormap="gray",
            translate=tuple(origen_region),
            contrast_limits=tuple(imagen.contrast_limits),
        )
        viewer.layers.selection.active = capa

        antes = medidas_restauracion(region)
        despues = medidas_restauracion(restaurada)

        parametros = {"psf": tipo_psf}
        parametros.update({
            k: v for k, v in dict(
                sigma_psf=sigma_psf, radio_psf=radio_psf,
                longitud_psf=longitud_psf, angulo_psf=angulo_psf,
            ).items() if k in PARAMETROS_PSF[tipo_psf]
        })
        if tipo_psf == "Capa de napari":
            parametros["capa_psf"] = capa_psf.name
        parametros["psf_tamano"] = f"{psf.shape[0]}x{psf.shape[1]}"
        for clave in PARAMETROS_DECONVOLUCION[metodo]:
            parametros[clave] = {"balance": balance, "iteraciones": iteraciones,
                                 "tolerancia": tolerancia}[clave]

        historial = info.get("historial") or []
        resultados = {"tiempo_s": segundos, "capa_resultado": nombre_capa}
        detalle = []
        if historial:
            ultimo = historial[-1]
            resultados.update(
                iteraciones_realizadas=ultimo["iteracion"],
                residuo_rms_final=ultimo["residuo_rms"],
                cambio_final=ultimo["cambio"],
            )
            detalle.append(
                f"Iteraciones realizadas: {ultimo['iteracion']} de {iteraciones}"
                + ("  (cortó por tolerancia)" if tolerancia > 0
                   and ultimo["cambio"] < tolerancia else "")
            )
            detalle.append(
                f"Residuo RMS: {historial[0]['residuo_rms']:.4f} → "
                f"{ultimo['residuo_rms']:.4f}   "
                f"cambio relativo final: {ultimo['cambio']:.5f}"
            )
        elif info.get("balance_usado") is not None:
            resultados["balance_usado"] = info["balance_usado"]
        else:
            resultados["balance_usado"] = "estimado (no supervisado)"

        if ver_figura:
            nombre_fig = f"Deconvolución: {imagen.name} - {metodo}"
            fig = figura_deconvolucion(
                f"{nombre_fig} ({formatear_resultados(parametros)})",
                region, restaurada, psf, historial,
            )
            mostrar_figura(fig, nombre_fig)

        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        if registrar:
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="4 - Restauración (deconvolución)",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=metodo,
                parametros=parametros,
                metrica_antes=antes,
                metrica_despues=despues,
                resultados=resultados,
            )])

        informe.value = (
            "INFORME - DECONVOLUCIÓN (Act. 4)\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Método: {metodo}\n"
            "Parámetros: " + formatear_resultados(parametros) + "\n"
            f"Tiempo: {segundos:.2f} s   Capa creada: {nombre_capa}\n"
            + ("".join(f"{d}\n" for d in detalle))
            + "\nANTES → DESPUÉS\n"
            + "\n".join(
                f"  {clave}: {formatear_valor(antes[clave])} → "
                f"{formatear_valor(despues[clave])}"
                for clave in antes
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo deconvolucionar:\n{error}"


def actualizar_parametros_deconv(event=None):
    """Muestra solo los parámetros de la PSF y del método elegidos."""
    visibles = set(PARAMETROS_PSF[widget_deconvolucion.tipo_psf.value])
    visibles |= set(PARAMETROS_DECONVOLUCION[widget_deconvolucion.metodo.value])
    for nombre in TODOS_PARAMETROS_DECONV:
        getattr(widget_deconvolucion, nombre).visible = nombre in visibles


widget_deconvolucion.metodo.changed.connect(actualizar_parametros_deconv)
widget_deconvolucion.tipo_psf.changed.connect(actualizar_parametros_deconv)


# ------------------------------------------------------------
# 6.5.b  Artefactos periódicos: picos de Fourier y filtro notch
# ------------------------------------------------------------

# Picos detectados en la última corrida (se eligen por número)
picos_detectados = {"picos": [], "forma": None, "imagen": "", "region": ""}


def texto_picos(picos):
    return "\n".join(
        f"  {i + 1}. período = {p['periodo_px']:.2f} px   "
        f"frecuencia = {p['frecuencia']:.4f} ciclos/px   "
        f"ángulo = {p['angulo_grados']:+.1f}°   "
        f"prominencia = {p['prominencia']:.2f}   "
        f"(fila {p['fila']}, columna {p['columna']})"
        for i, p in enumerate(picos)
    )


def figura_notch(nombre, original, filtrada, log_mag, mascara, picos, radio):
    """Espectro con los notches marcados, filtrada, residuo y máscara."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    alto, ancho = original.shape

    ax = ejes[0, 0]
    ax.imshow(log_mag, cmap="gray")
    centro_f, centro_c = alto // 2, ancho // 2
    for i, p in enumerate(picos):
        for fila, columna in (
            (p["fila"], p["columna"]),
            (2 * centro_f - p["fila"], 2 * centro_c - p["columna"]),
        ):
            ax.add_patch(
                Circle((columna, fila), radio, fill=False, color="red", linewidth=1)
            )
        ax.annotate(str(i + 1), (p["columna"], p["fila"]), color="red", fontsize=8)
    ax.set_title("log |FFT| con los notches aplicados")

    ax = ejes[0, 1]
    ax.imshow(mascara, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Máscara del filtro (0 = atenuado)")

    ax = ejes[1, 0]
    vmin, vmax = np.percentile(original, [0.5, 99.5])
    ax.imshow(filtrada, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Imagen filtrada")

    ax = ejes[1, 1]
    residuo = original - filtrada
    lim = max(np.percentile(np.abs(residuo), 99.5), 1e-9)
    im = ax.imshow(residuo, cmap="gray", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Residuo = original - filtrada (lo que se quitó)")

    for ax in ejes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(nombre)
    return fig


boton_detectar_picos = PushButton(text="Detectar picos en Fourier")


@boton_detectar_picos.clicked.connect
def detectar_picos():
    try:
        imagen = widget_notch.imagen.value
        if imagen is None:
            raise ValueError("Elegí la capa en el bloque de artefactos.")
        region, _, descripcion = region_en_gris(
            imagen, widget_notch.modo.value, widget_notch.roi.value,
            widget_notch.excluir_marco.value,
        )
        picos, log_mag = detectar_picos_fft(
            region, cantidad=widget_notch.cantidad_picos.value
        )
        picos_detectados.update(
            picos=picos, forma=region.shape, imagen=imagen.name, region=descripcion
        )

        informe.value = (
            "INFORME - PICOS DE FOURIER (Act. 4)\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"{len(picos)} pico(s) más prominentes (cada uno con su simétrico):\n"
            + texto_picos(picos)
            + "\n\nPara filtrar, escribí sus números en 'Picos a filtrar'."
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudieron detectar picos:\n{error}"


@magicgui(
    call_button="Aplicar filtro notch",
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a filtrar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    cantidad_picos={"label": "Picos a detectar", "min": 1, "max": 20},
    picos_elegidos={"label": "Picos a filtrar (ej: 1,2)"},
    radio_notch={"label": "Radio del notch (px)", "min": 0.5, "max": 100.0, "step": 0.5},
    tipo_notch={"choices": TIPOS_NOTCH, "label": "Tipo de notch"},
    orden_butterworth={"label": "Orden (Butterworth)", "min": 1, "max": 10},
    ver_figura={"label": "Mostrar figura comparativa"},
    registrar={"label": "Agregar al CSV"},
)
def widget_notch(
    imagen: Image,
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    cantidad_picos: int = 6,
    picos_elegidos: str = "1",
    radio_notch: float = 5.0,
    tipo_notch: str = "Gaussiano",
    orden_butterworth: int = 2,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )

        if not picos_detectados["picos"]:
            raise ValueError("Primero tocá 'Detectar picos en Fourier'.")
        if picos_detectados["forma"] != region.shape:
            raise ValueError(
                "Los picos se detectaron sobre una región de otro tamaño "
                f"({picos_detectados['forma']}): detectalos de nuevo."
            )

        try:
            numeros = [int(n) for n in picos_elegidos.replace(" ", "").split(",") if n]
        except ValueError:
            raise ValueError("Escribí los picos como números separados por comas.")
        disponibles = picos_detectados["picos"]
        if not numeros or any(n < 1 or n > len(disponibles) for n in numeros):
            raise ValueError(f"Elegí números entre 1 y {len(disponibles)}.")
        picos = [disponibles[n - 1] for n in numeros]

        mascara = mascara_notch(
            region.shape, picos, radio_notch, tipo_notch, orden_butterworth
        )
        filtrada, espectro = aplicar_notch(region, mascara)
        log_mag = np.log1p(np.abs(espectro))

        nombre_capa = f"{imagen.name}_notch"
        capa = agregar_o_actualizar_capa(
            nombre_capa, filtrada, colormap="gray",
            translate=tuple(origen_region),
            contrast_limits=tuple(imagen.contrast_limits),
        )
        viewer.layers.selection.active = capa

        control = verificar_notch(region, filtrada)
        antes = medidas_restauracion(region)
        despues = medidas_restauracion(filtrada)
        picos_despues, _ = detectar_picos_fft(filtrada, cantidad=1)
        if picos_despues:
            despues["pico_fft_prominencia"] = picos_despues[0]["prominencia"]
        antes["pico_fft_prominencia"] = max(p["prominencia"] for p in picos)

        parametros = {
            "picos": ",".join(str(n) for n in numeros),
            "radio": radio_notch,
            "tipo": tipo_notch,
        }
        if tipo_notch == "Butterworth":
            parametros["orden"] = orden_butterworth
        for i, p in zip(numeros, picos):
            parametros[f"pico{i}_periodo_px"] = p["periodo_px"]
            parametros[f"pico{i}_angulo"] = p["angulo_grados"]

        if ver_figura:
            nombre_fig = f"Notch: {imagen.name}"
            fig = figura_notch(
                f"{nombre_fig} ({formatear_resultados(parametros)})",
                region, filtrada, log_mag, mascara, picos, radio_notch,
            )
            mostrar_figura(fig, nombre_fig)

        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        resultados = dict(control)
        resultados["capa_resultado"] = nombre_capa
        if registrar:
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="4 - Artefacto periódico (notch)",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=f"Notch {tipo_notch}",
                parametros=parametros,
                metrica_antes=antes,
                metrica_despues=despues,
                resultados=resultados,
            )])

        informe.value = (
            "INFORME - FILTRO NOTCH (Act. 4)\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Notch {tipo_notch}, radio {radio_notch:g} px, picos {parametros['picos']}\n"
            + texto_picos(picos)
            + f"\n\nCapa creada: {nombre_capa}\n"
            "\nANTES → DESPUÉS\n"
            + "\n".join(
                f"  {clave}: {formatear_valor(antes[clave])} → "
                f"{formatear_valor(despues.get(clave, float('nan')))}"
                for clave in antes
            )
            + "\n\nCONTROL DE LO REMOVIDO\n"
            + "\n".join(
                f"  {clave} = {formatear_valor(valor)}"
                for clave, valor in control.items()
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo aplicar el notch:\n{error}"


# ============================================================
# 6.6 CORRECCIÓN DE FONDO (Actividad 5)
#     Cinco métodos con sus parámetros, más un barrido que genera de
#     una vez subcorrección / corrección adecuada / sobrecorrección.
# ============================================================

PARAMETROS_FONDO = {
    "Dark-field / flat-field": ["capa_flat", "capa_dark"],
    "Kernel de gran escala": [
        "sigma_fondo", "tipo_kernel", "modo_correccion", "conservar_nivel",
    ],
    "Rolling ball": ["radio_bola", "conservar_nivel"],
    "White top-hat": ["radio_tophat", "conservar_nivel"],
    "Homomórfica": ["gamma_bajo", "gamma_alto", "corte"],
}
TODOS_PARAMETROS_FONDO = sorted(
    {n for v in PARAMETROS_FONDO.values() for n in v}
)

# Parámetro que se barre en la comparación y en qué sentido actúa:
#   "suaviza"  -> más grande = corrección más suave (subcorrige)
#   "fortalece" -> más grande = corrección más fuerte (sobrecorrige)
PARAMETRO_BARRIDO = {
    "Kernel de gran escala": ("sigma_fondo", "suaviza"),
    "Rolling ball": ("radio_bola", "suaviza"),
    "White top-hat": ("radio_tophat", "suaviza"),
    "Homomórfica": ("corte", "fortalece"),
}


def figura_fondo(nombre, original, corregida, fondo):
    """Original, fondo estimado, corregida y perfil de la fila central."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    vmin, vmax = np.percentile(original, [0.5, 99.5])

    ax = ejes[0, 0]
    ax.imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Original")

    ax = ejes[0, 1]
    im = ax.imshow(fondo, cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Fondo estimado")

    ax = ejes[1, 0]
    ax.imshow(corregida, cmap="gray")
    ax.set_title("Corregida (contraste propio)")

    for ax in (ejes[0, 0], ejes[0, 1], ejes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])

    ax = ejes[1, 1]
    fila = original.shape[0] // 2
    ax.plot(original[fila], color="0.6", linewidth=0.8, label="original")
    ax.plot(fondo[fila], color="tab:green", linewidth=1.0, label="fondo")
    ax.plot(corregida[fila], color="tab:red", linewidth=1.0, label="corregida")
    ax.set_xlabel("columna (px)")
    ax.set_title(f"Perfil horizontal (fila central {fila})")
    ax.legend(fontsize=7)

    fig.suptitle(nombre)
    return fig


def figura_barrido(nombre, original, corregidas, parametro):
    """
    Compara subcorrección / adecuada / sobrecorrección.
    `corregidas` es [(etiqueta, valor, imagen), ...].
    """
    fig = Figure(figsize=(15, 9), layout="constrained")
    ejes = fig.subplots(2, 3)
    vmin, vmax = np.percentile(original, [0.5, 99.5])

    ax = ejes[0, 0]
    ax.imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Original")

    for ax, (etiqueta, valor, imagen) in zip(
        [ejes[0, 1], ejes[0, 2], ejes[1, 0]], corregidas
    ):
        ax.imshow(imagen, cmap="gray")
        ax.set_title(f"{etiqueta} ({parametro} = {valor:g})")

    for ax in (ejes[0, 0], ejes[0, 1], ejes[0, 2], ejes[1, 0]):
        ax.set_xticks([])
        ax.set_yticks([])

    fila = original.shape[0] // 2
    ax = ejes[1, 1]
    for etiqueta, valor, imagen in corregidas:
        suave = ndi.gaussian_filter(imagen, sigma=max(3, min(imagen.shape) / 20))
        ax.plot(suave[fila], linewidth=1.0, label=f"{etiqueta} ({valor:g})")
    suave_original = ndi.gaussian_filter(
        original, sigma=max(3, min(original.shape) / 20)
    )
    ax.plot(suave_original[fila], color="0.6", linewidth=0.8, label="original")
    ax.set_xlabel("columna (px)")
    ax.set_title("Fondo residual (imagen muy suavizada, fila central)")
    ax.legend(fontsize=7)

    ax = ejes[1, 2]
    bins = 80
    ax.hist(original.ravel(), bins=bins, density=True, color="0.6",
            histtype="step", label="original")
    for etiqueta, valor, imagen in corregidas:
        ax.hist(imagen.ravel(), bins=bins, density=True, histtype="step",
                linewidth=1.0, label=etiqueta)
    ax.set_yscale("log")
    ax.set_title("Histogramas")
    ax.legend(fontsize=7)

    fig.suptitle(nombre)
    return fig


def rois_en_arreglo(arreglo, origen_arreglo, roi_layer):
    """
    (indice, píxeles) de las ROIs que caen enteras dentro de `arreglo`,
    cuyo píxel [0, 0] está en `origen_arreglo` (coordenadas del mundo).
    Permite medir las mismas ROIs antes y después de procesar.
    """
    if roi_layer is None or len(roi_layer.data) == 0:
        return []

    desplazamiento = np.asarray(origen_arreglo, dtype=float)
    alto, ancho = arreglo.shape
    salida = []
    for indice in range(len(roi_layer.data)):
        vertices = np.asarray(roi_layer.data[indice], dtype=float) - desplazamiento
        if (
            vertices.min() < 0
            or vertices[:, 0].max() > alto
            or vertices[:, 1].max() > ancho
        ):
            continue
        filas, columnas = polygon(vertices[:, 0], vertices[:, 1], shape=arreglo.shape)
        if filas.size > 1:
            salida.append((indice, arreglo[filas, columnas]))
    return salida


def _datos_fondo(imagen, modo, roi, excluir_marco):
    """Región a corregir, ROIs que caen adentro y descripción."""
    region, origen_region, descripcion = region_en_gris(
        imagen, modo, roi, excluir_marco
    )
    return region, origen_region, descripcion, rois_en_arreglo(
        region, origen_region, roi
    )


@magicgui(
    call_button="Corregir fondo",
    metodo={"choices": METODOS_FONDO, "label": "Método"},
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a corregir",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    capa_flat={"label": "Capa flat-field"},
    capa_dark={"label": "Capa dark-field (opcional)"},
    sigma_fondo={"label": "Tamaño del kernel (px)", "min": 1.0, "max": 500.0, "step": 5.0},
    tipo_kernel={"choices": ["Gaussiano", "Mediana"], "label": "Kernel"},
    modo_correccion={"choices": ["Resta", "División"], "label": "Fondo aditivo/multiplicativo"},
    radio_bola={"label": "Radio de la bola (px)", "min": 1.0, "max": 500.0, "step": 5.0},
    radio_tophat={"label": "Radio del top-hat (px)", "min": 1.0, "max": 300.0, "step": 5.0},
    gamma_bajo={"label": "γ bajas frecuencias", "min": 0.0, "max": 2.0, "step": 0.05},
    gamma_alto={"label": "γ altas frecuencias", "min": 0.1, "max": 5.0, "step": 0.05},
    corte={"label": "Corte D0 (frac. Nyquist)", "min": 0.005, "max": 1.0, "step": 0.005},
    conservar_nivel={"label": "Conservar nivel medio"},
    factor_barrido={"label": "Factor del barrido", "min": 1.1, "max": 10.0, "step": 0.1},
    ver_figura={"label": "Mostrar figura"},
    registrar={"label": "Agregar al CSV"},
)
def widget_fondo(
    imagen: Image,
    metodo: str = "Rolling ball",
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    capa_flat: Image = None,
    capa_dark: Image = None,
    sigma_fondo: float = 50.0,
    tipo_kernel: str = "Gaussiano",
    modo_correccion: str = "Resta",
    radio_bola: float = 50.0,
    radio_tophat: float = 25.0,
    gamma_bajo: float = 0.5,
    gamma_alto: float = 1.5,
    corte: float = 0.05,
    conservar_nivel: bool = True,
    factor_barrido: float = 2.0,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion, lista_rois = _datos_fondo(
            imagen, modo, roi, excluir_marco
        )

        inicio = datetime.now()
        corregida, fondo = corregir_fondo(
            region, metodo=metodo,
            flat=capa_flat.data if capa_flat is not None else None,
            dark=capa_dark.data if capa_dark is not None else None,
            sigma_fondo=sigma_fondo, tipo_kernel=tipo_kernel,
            modo_correccion=modo_correccion, radio_bola=radio_bola,
            radio_tophat=radio_tophat, gamma_bajo=gamma_bajo,
            gamma_alto=gamma_alto, corte=corte, conservar_nivel=conservar_nivel,
        )
        segundos = (datetime.now() - inicio).total_seconds()

        nombre_capa = f"{imagen.name}_{clave_csv(metodo)}"
        capa = agregar_o_actualizar_capa(
            nombre_capa, corregida, colormap="gray",
            translate=tuple(origen_region),
        )
        agregar_o_actualizar_capa(
            f"{nombre_capa}_fondo", fondo, colormap="viridis",
            translate=tuple(origen_region), visible=False,
        )
        viewer.layers.selection.active = capa

        antes = medidas_fondo(region, lista_rois)
        despues = medidas_fondo(
            corregida, rois_en_arreglo(corregida, origen_region, roi)
        )

        parametros = {
            clave: valor for clave, valor in dict(
                capa_flat=capa_flat.name if capa_flat is not None else "",
                capa_dark=capa_dark.name if capa_dark is not None else "",
                sigma_fondo=sigma_fondo, tipo_kernel=tipo_kernel,
                modo_correccion=modo_correccion, radio_bola=radio_bola,
                radio_tophat=radio_tophat, gamma_bajo=gamma_bajo,
                gamma_alto=gamma_alto, corte=corte,
                conservar_nivel=conservar_nivel,
            ).items() if clave in PARAMETROS_FONDO[metodo]
        }

        if ver_figura:
            nombre_fig = f"Fondo: {imagen.name} - {metodo}"
            fig = figura_fondo(
                f"{nombre_fig} ({formatear_resultados(parametros)})",
                region, corregida, fondo,
            )
            mostrar_figura(fig, nombre_fig)

        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        if registrar:
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="5 - Corrección de fondo",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=metodo,
                parametros=parametros,
                metrica_antes=antes,
                metrica_despues=despues,
                resultados={"tiempo_s": segundos, "capa_resultado": nombre_capa},
            )])

        informe.value = (
            "INFORME - CORRECCIÓN DE FONDO (Act. 5)\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Método: {metodo}\n"
            "Parámetros: " + formatear_resultados(parametros) + "\n"
            f"Tiempo: {segundos:.2f} s   Capas creadas: {nombre_capa} "
            f"(+ _fondo)\n\n"
            "ANTES → DESPUÉS\n"
            + "\n".join(
                f"  {clave}: {formatear_valor(antes[clave])} → "
                f"{formatear_valor(despues.get(clave, float('nan')))}"
                for clave in antes
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo corregir el fondo:\n{error}"


boton_barrido_fondo = PushButton(
    text="Crear sub / adecuada / sobrecorrección"
)


@boton_barrido_fondo.clicked.connect
def barrido_fondo():
    """
    Corre el método tres veces: con el parámetro del panel y con ese
    parámetro dividido y multiplicado por el factor. Crea las tres capas,
    una figura comparativa y tres filas del CSV.
    """
    try:
        valores = widget_fondo.asdict()
        metodo = valores["metodo"]
        imagen = valores["imagen"]
        if imagen is None:
            raise ValueError("Elegí la capa en el bloque de corrección de fondo.")
        if metodo not in PARAMETRO_BARRIDO:
            raise ValueError(
                "El barrido no aplica a dark/flat-field: la corrección queda "
                "definida por los campos de referencia, no por un parámetro."
            )

        parametro, sentido = PARAMETRO_BARRIDO[metodo]
        base = valores[parametro]
        factor = valores["factor_barrido"]
        menor, mayor = base / factor, base * factor
        if sentido == "suaviza":
            barrido = [
                ("Sobrecorrección", menor),
                ("Corrección adecuada", base),
                ("Subcorrección", mayor),
            ]
        else:
            barrido = [
                ("Subcorrección", menor),
                ("Corrección adecuada", base),
                ("Sobrecorrección", mayor),
            ]

        region, origen_region, descripcion, lista_rois = _datos_fondo(
            imagen, valores["modo"], valores["roi"], valores["excluir_marco"]
        )
        antes = medidas_fondo(region, lista_rois)

        comunes = dict(
            flat=valores["capa_flat"].data if valores["capa_flat"] is not None else None,
            dark=valores["capa_dark"].data if valores["capa_dark"] is not None else None,
            sigma_fondo=valores["sigma_fondo"], tipo_kernel=valores["tipo_kernel"],
            modo_correccion=valores["modo_correccion"], radio_bola=valores["radio_bola"],
            radio_tophat=valores["radio_tophat"], gamma_bajo=valores["gamma_bajo"],
            gamma_alto=valores["gamma_alto"], corte=valores["corte"],
            conservar_nivel=valores["conservar_nivel"],
        )

        corregidas = []
        filas = []
        lineas = []
        region_fila, etiqueta = region_y_etiqueta(valores["modo"], valores["roi"])
        for nombre_caso, valor in barrido:
            argumentos = dict(comunes)
            argumentos[parametro] = valor
            corregida, _ = corregir_fondo(region, metodo=metodo, **argumentos)
            corregidas.append((nombre_caso, valor, corregida))

            nombre_capa = f"{imagen.name}_{clave_csv(metodo)}_{clave_csv(nombre_caso)}"
            agregar_o_actualizar_capa(
                nombre_capa, corregida, colormap="gray",
                translate=tuple(origen_region),
                visible=(nombre_caso == "Corrección adecuada"),
            )
            despues = medidas_fondo(
                corregida, rois_en_arreglo(corregida, origen_region, valores["roi"])
            )
            lineas.append(
                f"{nombre_caso} ({parametro} = {valor:g}) → capa {nombre_capa}\n"
                + "\n".join(
                    f"    {clave}: {formatear_valor(antes[clave])} → "
                    f"{formatear_valor(despues.get(clave, float('nan')))}"
                    for clave in antes
                )
            )
            filas.append(crear_registro(
                imagen=imagen.name,
                actividad="5 - Corrección de fondo",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=metodo,
                parametros={parametro: valor, "factor_barrido": factor},
                metrica_antes=antes,
                metrica_despues=despues,
                resultados={"capa_resultado": nombre_capa},
                observaciones=f"barrido: {nombre_caso.lower()}",
            ))

        if valores["registrar"]:
            agregar_pendientes(filas)
        if valores["ver_figura"]:
            nombre_fig = f"Barrido de fondo: {imagen.name} - {metodo}"
            mostrar_figura(
                figura_barrido(nombre_fig, region, corregidas, parametro), nombre_fig
            )

        informe.value = (
            "INFORME - BARRIDO DE CORRECCIÓN DE FONDO (Act. 5)\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Método: {metodo}   Parámetro barrido: {parametro} "
            f"(x1/{factor:g}, x1, x{factor:g})\n\n"
            + "\n\n".join(lineas)
            + (f"\n\n{len(filas)} filas agregadas al CSV pendiente."
               if valores["registrar"] else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo hacer el barrido:\n{error}"


def actualizar_parametros_fondo(event=None):
    """Muestra solo los parámetros del método de fondo elegido."""
    visibles = PARAMETROS_FONDO[widget_fondo.metodo.value]
    for nombre in TODOS_PARAMETROS_FONDO:
        getattr(widget_fondo, nombre).visible = nombre in visibles
    widget_fondo.factor_barrido.visible = (
        widget_fondo.metodo.value in PARAMETRO_BARRIDO
    )


widget_fondo.metodo.changed.connect(actualizar_parametros_fondo)


# ============================================================
# 6.7 REALCE (Actividad 6)
#     Transformaciones de intensidad y filtros de realce, con los
#     valores que permiten ver si se amplificó ruido o se perdió
#     información (la lectura queda para el informe).
# ============================================================

PARAMETROS_REALCE = {
    "Gamma": ["gamma"],
    "Logarítmica": ["factor_log"],
    "Sigmoidal": ["centro", "ganancia"],
    "CLAHE": ["tamano_mosaico", "limite_contraste"],
    "Pasa-altos": ["sigma_altos", "conservar_nivel"],
    "Pasa-banda": ["sigma_bajo", "sigma_alto", "conservar_nivel"],
    "Unsharp masking": ["sigma_unsharp", "cantidad", "umbral"],
}
TODOS_PARAMETROS_REALCE = sorted(
    {n for v in PARAMETROS_REALCE.values() for n in v}
)


def figura_realce(nombre, original, realzada, limites):
    """Original, realzada, histogramas y curva de transformación."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)

    ax = ejes[0, 0]
    ax.imshow(original, cmap="gray", vmin=limites[0], vmax=limites[1])
    ax.set_title("Original")

    ax = ejes[0, 1]
    ax.imshow(realzada, cmap="gray")
    ax.set_title("Realzada (contraste propio)")

    for ax in (ejes[0, 0], ejes[0, 1]):
        ax.set_xticks([])
        ax.set_yticks([])

    ax = ejes[1, 0]
    ax.hist(original.ravel(), bins=128, density=True, color="0.6",
            histtype="step", label="original")
    ax.hist(realzada.ravel(), bins=128, density=True, color="tab:red",
            histtype="step", label="realzada")
    ax.set_yscale("log")
    ax.set_title("Histogramas")
    ax.set_xlabel("intensidad")
    ax.legend(fontsize=7)

    # Relación entrada-salida: para las transformaciones puntuales es la
    # curva del método; para los filtros muestra cuánta dispersión agrega.
    ax = ejes[1, 1]
    muestra = np.random.default_rng(0).choice(original.size, size=min(20000, original.size),
                                              replace=False)
    ax.plot(original.ravel()[muestra], realzada.ravel()[muestra], ".",
            markersize=1, alpha=0.3)
    ax.set_xlabel("intensidad original")
    ax.set_ylabel("intensidad realzada")
    ax.set_title("Entrada vs. salida")

    fig.suptitle(nombre)
    return fig


@magicgui(
    call_button="Aplicar realce",
    metodo={"choices": METODOS_REALCE, "label": "Método"},
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a realzar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    gamma={"label": "γ", "min": 0.05, "max": 5.0, "step": 0.05},
    factor_log={"label": "Factor del log", "min": 0.1, "max": 500.0, "step": 1.0},
    centro={"label": "Centro (frac. del rango)", "min": 0.0, "max": 1.0, "step": 0.05},
    ganancia={"label": "Ganancia (pendiente)", "min": 0.5, "max": 50.0, "step": 0.5},
    tamano_mosaico={"label": "Mosaico CLAHE (px)", "min": 3, "max": 512},
    limite_contraste={"label": "Límite de contraste", "min": 0.001, "max": 1.0, "step": 0.005},
    sigma_altos={"label": "σ del pasa-altos (px)", "min": 0.5, "max": 100.0, "step": 0.5},
    sigma_bajo={"label": "σ bajo (px)", "min": 0.3, "max": 50.0, "step": 0.1},
    sigma_alto={"label": "σ alto (px)", "min": 0.5, "max": 200.0, "step": 0.5},
    sigma_unsharp={"label": "σ del unsharp (px)", "min": 0.3, "max": 50.0, "step": 0.1},
    cantidad={"label": "Cantidad", "min": 0.1, "max": 10.0, "step": 0.1},
    umbral={"label": "Umbral de detalle", "min": 0.0, "max": 100.0, "step": 0.5},
    conservar_nivel={"label": "Conservar nivel medio"},
    ver_figura={"label": "Mostrar figura"},
    registrar={"label": "Agregar al CSV"},
)
def widget_realce(
    imagen: Image,
    metodo: str = "CLAHE",
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    gamma: float = 0.7,
    factor_log: float = 10.0,
    centro: float = 0.5,
    ganancia: float = 10.0,
    tamano_mosaico: int = 64,
    limite_contraste: float = 0.01,
    sigma_altos: float = 3.0,
    sigma_bajo: float = 1.0,
    sigma_alto: float = 8.0,
    sigma_unsharp: float = 2.0,
    cantidad: float = 1.0,
    umbral: float = 0.0,
    conservar_nivel: bool = True,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        rois_antes = rois_en_arreglo(region, origen_region, roi)

        todos = dict(
            gamma=gamma, factor_log=factor_log, centro=centro, ganancia=ganancia,
            tamano_mosaico=tamano_mosaico, limite_contraste=limite_contraste,
            sigma_altos=sigma_altos, sigma_bajo=sigma_bajo, sigma_alto=sigma_alto,
            sigma_unsharp=sigma_unsharp, cantidad=cantidad, umbral=umbral,
            conservar_nivel=conservar_nivel,
        )
        parametros = {k: todos[k] for k in PARAMETROS_REALCE[metodo]}

        inicio = datetime.now()
        realzada = realzar(region, metodo=metodo, **todos)
        segundos = (datetime.now() - inicio).total_seconds()

        nombre_capa = f"{imagen.name}_{clave_csv(metodo)}"
        capa = agregar_o_actualizar_capa(
            nombre_capa, realzada, colormap="gray",
            translate=tuple(origen_region),
        )
        viewer.layers.selection.active = capa

        limites = (float(region.min()), float(region.max()))
        antes = medidas_realce(region, rois_antes, limites)
        despues = medidas_realce(
            realzada, rois_en_arreglo(realzada, origen_region, roi), limites
        )
        razones = comparar_realce(region, realzada)

        if ver_figura:
            nombre_fig = f"Realce: {imagen.name} - {metodo}"
            fig = figura_realce(
                f"{nombre_fig} ({formatear_resultados(parametros)})",
                region, realzada, limites,
            )
            mostrar_figura(fig, nombre_fig)

        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        if registrar:
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="6 - Realce",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=metodo,
                parametros=parametros,
                metrica_antes=antes,
                metrica_despues=despues,
                resultados={**razones, "tiempo_s": segundos,
                            "capa_resultado": nombre_capa},
            )])

        informe.value = (
            "INFORME - REALCE (Act. 6)\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"Método: {metodo}\n"
            "Parámetros: " + formatear_resultados(parametros) + "\n"
            f"Tiempo: {segundos:.2f} s   Capa creada: {nombre_capa}\n\n"
            "ANTES → DESPUÉS\n"
            + "\n".join(
                f"  {clave}: {formatear_valor(antes[clave])} → "
                f"{formatear_valor(despues.get(clave, float('nan')))}"
                for clave in antes
            )
            + "\n\nRAZONES DESPUÉS/ANTES\n"
            + "\n".join(
                f"  {clave} = {formatear_valor(valor)}"
                for clave, valor in razones.items()
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo realzar:\n{error}"


def actualizar_parametros_realce(event=None):
    """Muestra solo los parámetros del método de realce elegido."""
    visibles = PARAMETROS_REALCE[widget_realce.metodo.value]
    for nombre in TODOS_PARAMETROS_REALCE:
        getattr(widget_realce, nombre).visible = nombre in visibles


widget_realce.metodo.changed.connect(actualizar_parametros_realce)


# ============================================================
# 6.8 EVALUACIÓN (Actividad 7)
#     Métricas sin referencia (SNR, CNR, uniformidad) sobre las ROIs y,
#     si se elige una capa de referencia, también MSE / PSNR / SSIM.
# ============================================================

def figura_metricas(nombre, imagen, referencia, mapa_ssim, sin_referencia):
    """Imagen (y referencia), diferencia, mapa SSIM y barras de SNR/CNR."""
    fig = Figure(figsize=(12, 9), layout="constrained")
    ejes = fig.subplots(2, 2)
    vmin, vmax = np.percentile(imagen, [0.5, 99.5])

    ax = ejes[0, 0]
    ax.imshow(imagen, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Imagen evaluada")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = ejes[0, 1]
    if referencia is not None:
        ax.imshow(referencia, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title("Referencia")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.hist(imagen.ravel(), bins=128, color="0.5")
        ax.set_yscale("log")
        ax.set_title("Histograma de la imagen evaluada")

    ax = ejes[1, 0]
    if referencia is not None:
        diferencia = imagen - referencia
        lim = max(np.percentile(np.abs(diferencia), 99), 1e-9)
        im = ax.imshow(diferencia, cmap="gray", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Diferencia (evaluada - referencia)")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis("off")

    ax = ejes[1, 1]
    if mapa_ssim is not None:
        im = ax.imshow(mapa_ssim, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Mapa SSIM (1 = igual a la referencia)")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        claves = [
            c for c in sin_referencia
            if c.startswith(("snr_roi", "cnr_roi", "uniformidad_roi"))
        ]
        if claves:
            valores = [sin_referencia[c] for c in claves]
            ax.barh(range(len(claves)), valores, color="0.5")
            ax.set_yticks(range(len(claves)))
            ax.set_yticklabels(claves, fontsize=7)
            ax.invert_yaxis()
            ax.set_title("Métricas por ROI")
        else:
            ax.text(0.5, 0.5, "Dibujá ROIs para calcular\nSNR, CNR y uniformidad",
                    ha="center", va="center")
            ax.axis("off")

    fig.suptitle(nombre)
    return fig


@magicgui(
    call_button="Calcular métricas",
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a evaluar",
    },
    excluir_marco={"label": "Quitar marco uniforme"},
    referencia={"label": "Referencia (opcional)"},
    ver_figura={"label": "Mostrar figura"},
    registrar={"label": "Agregar al CSV"},
)
def widget_metricas(
    imagen: Image,
    referencia: Image = None,
    modo: str = "Imagen completa",
    roi: Shapes = None,
    excluir_marco: bool = True,
    ver_figura: bool = True,
    registrar: bool = True,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        region, origen_region, descripcion = region_en_gris(
            imagen, modo, roi, excluir_marco
        )
        lista_rois = rois_en_arreglo(region, origen_region, roi)

        # Misma región en la referencia (se alinea por las coordenadas del mundo)
        region_referencia = None
        if referencia is not None:
            if referencia is imagen:
                raise ValueError("La referencia es la misma capa que la evaluada.")
            gris_referencia = a_escala_de_grises(referencia.data)
            f0, c0 = np.round(origen_region - origen_capa(referencia)).astype(int)
            alto, ancho = region.shape
            if (
                f0 < 0 or c0 < 0
                or f0 + alto > gris_referencia.shape[0]
                or c0 + ancho > gris_referencia.shape[1]
            ):
                raise ValueError(
                    f"La región no entra en la referencia '{referencia.name}' "
                    f"({gris_referencia.shape})."
                )
            region_referencia = gris_referencia[f0:f0 + alto, c0:c0 + ancho]

        # El rango dinámico sale de la REFERENCIA: así el PSNR y el SSIM
        # de distintas versiones procesadas son comparables entre sí.
        sin_referencia, con_referencia, mapa = evaluar(
            region,
            referencia=region_referencia,
            lista_rois=lista_rois,
            rango_datos=escala_intensidad(
                referencia.data if referencia is not None else imagen.data
            ),
        )

        if ver_figura:
            nombre_fig = f"Métricas: {imagen.name}"
            fig = figura_metricas(
                nombre_fig + (f" vs {referencia.name}" if referencia else ""),
                region, region_referencia, mapa, sin_referencia,
            )
            mostrar_figura(fig, nombre_fig)

        region_fila, etiqueta = region_y_etiqueta(modo, roi)
        if registrar:
            metricas_referencia = {}
            if region_referencia is not None:
                metricas_referencia = metricas_sin_referencia(
                    region_referencia,
                    rois_en_arreglo(region_referencia, origen_region, roi),
                )
            agregar_pendientes([crear_registro(
                imagen=imagen.name,
                actividad="7 - Evaluación",
                region=region_fila,
                etiqueta_roi=etiqueta,
                metodo=(
                    f"Con y sin referencia (vs {referencia.name})"
                    if referencia is not None else "Sin referencia"
                ),
                parametros={"sin_marco": excluir_marco,
                            "rois": len(lista_rois)},
                metrica_antes=metricas_referencia,
                metrica_despues=sin_referencia,
                resultados=con_referencia,
            )])

        informe.value = (
            "INFORME - EVALUACIÓN (Act. 7)\n"
            f"Capa: {imagen.name}   Región: {descripcion} "
            f"({region.shape[0]}x{region.shape[1]})\n"
            f"ROIs usadas: {len(lista_rois)}\n\n"
            "SIN REFERENCIA\n"
            + "\n".join(
                f"  {clave} = {formatear_valor(valor)}"
                for clave, valor in sin_referencia.items()
            )
            + (
                f"\n\nCON REFERENCIA (vs {referencia.name})\n"
                + "\n".join(
                    f"  {clave} = {formatear_valor(valor)}"
                    for clave, valor in con_referencia.items()
                )
                if con_referencia else
                "\n\n(Sin capa de referencia: solo métricas sin referencia.)"
            )
            + ("\n\nFila agregada al CSV pendiente." if registrar else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudieron calcular las métricas:\n{error}"


# ============================================================
# 7. REGISTRO DEL PIPELINE (Actividad 7 - documentación)
#    Carga un paso (diagnóstico/método/parámetros/métricas) al mismo CSV
#    que las mediciones. Los campos los completa el usuario (la hipótesis
#    sale del análisis propio); el CSV se guarda con el botón.
# ============================================================

@magicgui(
    call_button="Cargar paso al pipeline",
    diagnostico={
        "choices": [
            "Ruido",
            "Fondo / inhomogeneidad",
            "Artefacto estructurado",
            "Desenfoque / pérdida de resolución",
            "Sin degradación (referencia)",
        ],
        "label": "Diagnóstico (hipótesis)",
    },
)
def widget_pipeline(
    imagen: Image,
    diagnostico: str = "Ruido",
    metodo: str = "",
    parametros: str = "",
    estimacion_ruido: str = "",
    metrica_antes: str = "",
    metrica_despues: str = "",
    observaciones: str = "",
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    registro = crear_registro(
        imagen=imagen.name,
        actividad="Pipeline",
        diagnostico=diagnostico,
        metodo=metodo,
        parametros=parametros,
        estimacion_ruido=estimacion_ruido,
        metrica_antes=metrica_antes,
        metrica_despues=metrica_despues,
        observaciones=observaciones,
    )
    agregar_pendientes([registro])

    informe.value = (
        "INFORME\n"
        f"Paso preparado para: {imagen.name}\n"
        f"Diagnóstico: {diagnostico}\n"
        f"Método: {metodo or '(vacío)'}"
    )


# ============================================================
# 8. CONTROLES DEL CSV
# ============================================================

ruta_csv = LineEdit(
    value="resultados/resultados_tp1.csv",
    label="Archivo CSV",
)

boton_guardar_csv = PushButton(text="Guardar en CSV")
boton_descartar = PushButton(text="Descartar filas pendientes")


@boton_guardar_csv.clicked.connect
def guardar_resultados_actuales():
    try:
        cantidad = len(registros_pendientes)
        ruta = guardar_registros_csv(ruta_csv.value, registros_pendientes)
        registros_pendientes.clear()
        estado_csv.value = (
            "CSV\n"
            f"{cantidad} fila(s) agregada(s).\n"
            f"Archivo: {ruta}"
        )
    except Exception as error:
        estado_csv.value = f"CSV\nNo se pudo guardar:\n{error}"


@boton_descartar.clicked.connect
def descartar_pendientes():
    cantidad = len(registros_pendientes)
    registros_pendientes.clear()
    estado_csv.value = f"CSV\n{cantidad} fila(s) descartada(s)."


# ============================================================
# 9. PANEL LATERAL
# ============================================================

def titulo(texto):
    """Encabezado de sección del panel."""
    etiqueta = Label(value=texto)
    etiqueta.native.setStyleSheet("font-weight: bold; margin-top: 8px;")
    return etiqueta


# labels=False: sin la columna con los nombres internos de cada bloque
# (widget_gris, widget_pipeline...), que ensanchaba mucho el panel.
panel = Container(
    labels=False,
    widgets=[
        titulo("ROIs"),
        boton_crear_roi,
        etiqueta_nueva,
        boton_etiquetar,
        titulo("Escala de grises"),
        widget_gris,
        titulo("Medición (media / varianza / MAD)"),
        widget_medicion,
        titulo("Actividad 1 - Diagnóstico"),
        boton_crear_perfil,
        widget_diagnostico,
        titulo("Actividad 2 - Estimación de ruido"),
        widget_ruido,
        titulo("Actividad 3 - Reducción de ruido"),
        boton_usar_sigma,
        widget_filtro,
        titulo("Actividad 4 - Restauración (deconvolución)"),
        boton_nsr,
        widget_deconvolucion,
        titulo("Actividad 4 - Artefactos periódicos (notch)"),
        boton_detectar_picos,
        widget_notch,
        titulo("Actividad 5 - Corrección de fondo"),
        widget_fondo,
        boton_barrido_fondo,
        titulo("Actividad 6 - Realce"),
        widget_realce,
        titulo("Actividad 7 - Evaluación (métricas)"),
        widget_metricas,
        titulo("Registro del pipeline"),
        widget_pipeline,
        titulo("Archivo CSV"),
        ruta_csv,
        boton_guardar_csv,
        boton_descartar,
        estado_csv,
    ]
)

# El panel va dentro de un QScrollArea: si no entra en la pantalla
# aparece una barra de desplazamiento y no se tapan los botones.
panel_con_scroll = QScrollArea()
panel_con_scroll.setWidgetResizable(True)
panel_con_scroll.setWidget(panel.native)
panel_con_scroll.setMinimumWidth(panel.native.sizeHint().width() + 30)

viewer.window.add_dock_widget(
    panel_con_scroll,
    area="right",
    name="TP1 - Preprocesamiento",
)

# El informe va en su propio panel fijo, abajo del visor: queda siempre
# a la vista sin importar cuánto se desplace el panel lateral.
viewer.window.add_dock_widget(
    informe.native,
    area="bottom",
    name="Informe",
)

# Al estar dentro del QScrollArea, magicgui no detecta solo el visor:
# se refrescan a mano las listas de capas (Image / Shapes) de los widgets.
viewer.layers.events.inserted.connect(panel.reset_choices)
viewer.layers.events.removed.connect(panel.reset_choices)
viewer.layers.events.moved.connect(panel.reset_choices)
panel.reset_choices()
actualizar_parametros_filtro()
actualizar_parametros_deconv()
actualizar_parametros_fondo()
actualizar_parametros_realce()


# ============================================================
# 10. INICIAR NAPARI
# ============================================================

if __name__ == "__main__":
    napari.run()

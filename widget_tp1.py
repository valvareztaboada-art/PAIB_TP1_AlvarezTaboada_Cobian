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
# ------------------------------------------------------------
# ESTADO: ACTIVIDADES 1, 2 Y 3 IMPLEMENTADAS.
#
# Lo que YA funciona:
#   - abrir napari con el panel del TP;
#   - convertir la imagen (o el recorte de una ROI) a escala de grises 2D;
#   - crear una capa de ROIs, etiquetar cada ROI con su segmento
#     (fondo, vaso, disco...) y medir media/varianza/desv/MAD en una
#     o en todas las ROIs;
#   - Actividad 1: diagnóstico (histogramas global/regional, perfiles,
#     variabilidad local, diferencias, FFT, autocorrelación, wavelet)
#     sobre la imagen completa o una ROI;
#   - Actividad 2: estimación de σ del ruido (ROI homogénea con media
#     observada o fija, múltiples ROIs, MAD, diferencias entre
#     adquisiciones, diferencias locales, wavelet);
#   - Actividad 3: reducción de ruido (gaussiano, bilateral, difusión
#     anisotrópica, NLM, TV, wavelet thresholding, Wiener local, BM3D)
#     con parámetros elegidos por el usuario; el resultado es una capa
#     nueva que puede seguir en el pipeline;
#   - todas las mediciones y los pasos del pipeline van al MISMO CSV
#     acumulativo, con la región y la etiqueta de la ROI (Act. 7).
#
# El widget solo calcula valores: la hipótesis de la degradación
# dominante, la justificación de la estimación y la elección de
# parámetros se hacen en el informe, a partir de esos valores.
#
# Lo que hay que COMPLETAR (marcado con  # >>> ACÁ COMPLETAMOS ):
#   - Actividad 4: restauración (deconvolución) y filtros notch
#   - Actividad 5: corrección de fondo
#   - Actividad 6: realce
#   - Actividad 7: métricas con y sin referencia
#
# Correr con:   pixi run python widget_tp1.py
# ============================================================

import csv
from pathlib import Path
from datetime import datetime

import cv2
import napari
import numpy as np
import pywt
from scipy import ndimage as ndi
from scipy.signal import wiener

from magicgui import magicgui
from magicgui.widgets import (
    Container,
    Label,
    PushButton,
    LineEdit,
    TextEdit,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from napari.layers import Image, Shapes
from qtpy.QtWidgets import QScrollArea, QVBoxLayout, QWidget
from skimage.draw import polygon
from skimage.measure import profile_line
from skimage.restoration import (
    denoise_nl_means,
    denoise_tv_chambolle,
    denoise_wavelet,
)


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
def restaurar(imagen, metodo="wiener", **kwargs):
    # >>> ACÁ COMPLETAMOS: deconvolución con PSF / notch para artefactos periódicos
    raise NotImplementedError("Actividad 4: restauración sin implementar.")


# ------------------------------------------------------------
# Actividad 5 - Corrección de fondo
# (dark/flat-field, kernel de gran escala, rolling ball,
#  white top-hat, homomórfica)
# ------------------------------------------------------------
def corregir_fondo(imagen, metodo="rolling_ball", **kwargs):
    # >>> ACÁ COMPLETAMOS: corrección de iluminación / inhomogeneidad
    raise NotImplementedError("Actividad 5: corrección de fondo sin implementar.")


# ------------------------------------------------------------
# Actividad 6 - Realce
# (gamma, log, sigmoidal, CLAHE, pasa-altos, pasa-banda, unsharp)
# ------------------------------------------------------------
def realzar(imagen, metodo="clahe", **kwargs):
    # >>> ACÁ COMPLETAMOS: realce de contraste sin amplificar ruido de más
    raise NotImplementedError("Actividad 6: realce sin implementar.")


# ------------------------------------------------------------
# Actividad 7 - Evaluación
# (métricas SIN referencia: SNR, CNR, etc. / CON referencia: PSNR, SSIM)
# ------------------------------------------------------------
def evaluar(imagen_antes, imagen_despues, referencia=None, **kwargs):
    # >>> ACÁ COMPLETAMOS: calcular las métricas que correspondan al objetivo
    raise NotImplementedError("Actividad 7: métricas sin implementar.")


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
                   widget_ruido, widget_filtro):
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
    """'Diferencias entre adquisiciones' -> 'diferencias_entre_adquisiciones'."""
    reemplazos = str.maketrans("áéíóúñ ", "aeioun_")
    return texto.lower().translate(reemplazos)


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


# ============================================================
# 10. INICIAR NAPARI
# ============================================================

if __name__ == "__main__":
    napari.run()

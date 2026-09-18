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
# ESTADO: ACTIVIDAD 1 IMPLEMENTADA.
#
# Lo que YA funciona:
#   - abrir napari con el panel del TP;
#   - convertir la imagen (o el recorte de una ROI) a escala de grises 2D;
#   - crear una capa de ROIs y medir media/varianza/desv/MAD (Act. 1 y 2);
#   - Actividad 1: diagnóstico (histogramas global/regional, perfiles,
#     variabilidad local, diferencias, FFT, autocorrelación, wavelet)
#     con una hipótesis sugerida de la degradación dominante;
#   - registrar cada paso del pipeline y exportar el CSV acumulativo (Act. 7).
#
# Lo que hay que COMPLETAR (marcado con  # >>> ACÁ COMPLETAMOS ):
#   - Actividad 2: estimación de ruido
#   - Actividad 3: filtros de reducción de ruido
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

import napari
import numpy as np
import pywt
from scipy import ndimage as ndi

from magicgui import magicgui
from magicgui.widgets import (
    Container,
    Label,
    PushButton,
    LineEdit,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from napari.layers import Image, Shapes
from qtpy.QtWidgets import QVBoxLayout, QWidget
from skimage.draw import polygon
from skimage.measure import profile_line


# ============================================================
# 1. HELPERS BASE (traídos de los widgets de la clase)
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
#    Columnas mínimas pedidas por el TP.
# ============================================================

COLUMNAS_CSV = [
    "fecha_hora",
    "imagen",
    "diagnostico",
    "metodo",
    "parametros",
    "estimacion_ruido",
    "metrica_antes",
    "metrica_despues",
    "observaciones",
]


def crear_registro(
    imagen="",
    diagnostico="",
    metodo="",
    parametros="",
    estimacion_ruido="",
    metrica_antes="",
    metrica_despues="",
    observaciones="",
):
    """Arma un paso del pipeline como una fila del CSV."""
    return {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "imagen": str(imagen),
        "diagnostico": str(diagnostico),
        "metodo": str(metodo),
        "parametros": str(parametros),
        "estimacion_ruido": str(estimacion_ruido),
        "metrica_antes": str(metrica_antes),
        "metrica_despues": str(metrica_despues),
        "observaciones": str(observaciones),
    }


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
                "El CSV existente tiene columnas distintas a las del widget."
            )

    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        if not existe:
            escritor.writeheader()
        escritor.writerows(registros)

    return ruta


# ============================================================
# 3. FUNCIONES DEL PIPELINE  ->  # >>> ACÁ COMPLETAMOS
#    Cada una recibe una imagen 2D (np.ndarray) y devuelve otra 2D.
#    Por ahora están vacías: lanzan NotImplementedError a propósito.
# ============================================================

# ------------------------------------------------------------
# Actividad 1 - Diagnóstico avanzado
# (histograma, perfiles, MAD, mapas de variabilidad local,
#  diferencias entre píxeles, FFT, autocorrelación, wavelet)
# ------------------------------------------------------------

# Umbrales orientativos para sugerir la hipótesis. Se calibraron
# mirando el conjunto Retina1..7 (sin marco blanco); no son universales.
UMBRAL_IMPULSOS = 0.002        # fracción de píxeles atípicos vs. mediana 3x3
UMBRAL_PICO_FFT = 3.0          # prominencia (log) de un pico fuera del centro
UMBRAL_RUIDO_RELATIVO = 0.03   # sigma_wavelet / rango dinámico (p99 - p1)
UMBRAL_FONDO = 0.15            # |1 - borde/centro| del fondo suavizado
UMBRAL_ALTAS_FRECUENCIAS = 0.19  # E(0.25-0.5 Nyq) / E(0.10-0.25 Nyq)


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


def sugerir_hipotesis(indicadores):
    """
    Sugerencia ORIENTATIVA de la degradación dominante a partir de los
    indicadores. La hipótesis final la formula quien analiza, mirando
    también los gráficos.
    Devuelve (categoria, lista_de_evidencias).
    """
    evidencias = []
    candidatas = []

    if indicadores["fraccion_impulsos"] > UMBRAL_IMPULSOS:
        candidatas.append("Ruido")
        evidencias.append(
            f"Ruido impulsivo: {100 * indicadores['fraccion_impulsos']:.2f}% "
            "de píxeles atípicos vs. mediana 3x3 "
            f"(curtosis de diferencias {indicadores['curtosis_diferencias']:.1f})."
        )

    if indicadores["pico_fft"] > UMBRAL_PICO_FFT:
        candidatas.append("Artefacto estructurado")
        evidencias.append(
            f"Pico en Fourier de prominencia {indicadores['pico_fft']:.2f} "
            f"(período ~{indicadores['pico_periodo']:.1f} px): patrón periódico."
        )

    if indicadores["ruido_relativo"] > UMBRAL_RUIDO_RELATIVO:
        candidatas.append("Ruido")
        evidencias.append(
            f"sigma_wavelet / rango = {indicadores['ruido_relativo']:.3f}: "
            "mucha energía en el detalle fino."
        )

    if abs(1 - indicadores["relacion_borde_centro"]) > UMBRAL_FONDO:
        candidatas.append("Fondo / inhomogeneidad")
        evidencias.append(
            "Fondo borde/centro = "
            f"{indicadores['relacion_borde_centro']:.2f}: iluminación no uniforme."
        )

    if (
        indicadores["relacion_altas"] < UMBRAL_ALTAS_FRECUENCIAS
        and indicadores["ruido_relativo"] <= UMBRAL_RUIDO_RELATIVO
    ):
        candidatas.append("Desenfoque / pérdida de resolución")
        evidencias.append(
            "Poca energía en frecuencias medias-altas "
            f"(relación {indicadores['relacion_altas']:.3f}): bordes suavizados."
        )

    if not candidatas:
        return "Sin degradación dominante evidente", [
            "Ningún indicador supera su umbral (¿imagen de referencia?)."
        ]

    return candidatas[0], evidencias


def diagnosticar(imagen, ventana=7, wavelet="db2", niveles=3, **kwargs):
    """
    Corre todos los análisis de la Actividad 1 sobre una imagen 2D
    (ya en escala de grises) y devuelve un diccionario con los
    indicadores, los mapas y la hipótesis sugerida.
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
        "fraccion_impulsos": fraccion_impulsos(img),
        "curtosis_diferencias": diferencias["curtosis"],
        "pico_fft": fourier["pico_prominencia"],
        "pico_periodo": fourier["pico_periodo"],
        "ruido_relativo": ondita["sigma_donoho"] / max(rango, 1e-12),
        "relacion_borde_centro": fondo["relacion_borde_centro"],
        "relacion_altas": fourier["relacion_altas"],
    }
    hipotesis, evidencias = sugerir_hipotesis(indicadores)

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
        "hipotesis": hipotesis,
        "evidencias": evidencias,
    }


# ------------------------------------------------------------
# Actividad 2 - Estimación de ruido
# (ROI homogénea, múltiples ROIs, MAD, diferencias, wavelet)
# ------------------------------------------------------------
def estimar_ruido(imagen, **kwargs):
    # >>> ACÁ COMPLETAMOS: estimar sigma del ruido y justificar el supuesto
    raise NotImplementedError("Actividad 2: estimación de ruido sin implementar.")


# ------------------------------------------------------------
# Actividad 3 - Reducción de ruido
# (gaussiano, bilateral, difusión anisotrópica, NLM, TV,
#  wavelet thresholding, Wiener local, BM3D)
# ------------------------------------------------------------
def reducir_ruido(imagen, metodo="gaussiano", **kwargs):
    # >>> ACÁ COMPLETAMOS: aplicar el filtro elegido según el diagnóstico
    raise NotImplementedError("Actividad 3: reducción de ruido sin implementar.")


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

informe = Label(value="INFORME\nTodavía no se realizó ningún cálculo.")
estado_csv = Label(value="CSV\nNo hay resultados pendientes.")


# ============================================================
# 5. CAPA DE ROIs (base de la clase, para Actividades 1 y 2)
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


def actualizar_colores_rois(event=None):
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


boton_crear_roi = PushButton(text="Crear capa de ROIs")


@boton_crear_roi.clicked.connect
def crear_capa_rois():
    roi_layer = obtener_capa_roi()

    if roi_layer is None:
        roi_layer = viewer.add_shapes(name="ROI", edge_width=3)
        roi_layer.events.data.connect(actualizar_colores_rois)
        mensaje = "Capa ROI creada.\nDibujá una o más regiones."
    else:
        mensaje = "La capa ROI ya existe.\nSe volvió a activar."

    viewer.layers.selection.active = roi_layer
    for widget in (widget_medicion, widget_gris, widget_diagnostico):
        widget.roi.value = roi_layer
    roi_layer.mode = "add_rectangle"

    informe.value = f"INFORME\n{mensaje}"


# ============================================================
# 6. HERRAMIENTA DE MEDICIÓN (Actividades 1 y 2)
#    Media / varianza / desvío sobre imagen completa o ROI.
# ============================================================

@magicgui(
    call_button="Medir estadísticas",
    modo={
        "choices": ["Imagen completa", "ROI seleccionada"],
        "label": "Región a analizar",
    },
)
def widget_medicion(
    imagen: Image,
    modo: str = "Imagen completa",
    roi: Shapes = None,
):
    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    try:
        img = a_escala_de_grises(imagen.data)

        if modo == "Imagen completa":
            datos = img
            region = "Imagen completa"
        else:
            if roi is None:
                raise ValueError("Creá primero una capa de ROIs.")
            indice = obtener_indice_roi_seleccionada(roi)
            datos = obtener_datos_roi(img, roi, indice, origen_capa(imagen))
            region = f"ROI {indice + 1} seleccionada"

        est = estadisticas_robustas(datos)

        informe.value = (
            "INFORME\n"
            f"Capa: {imagen.name}\n"
            f"Región: {region}\n"
            f"Número de píxeles: {est['n']}\n"
            f"Media: {est['media']:.3f}\n"
            f"Varianza muestral: {est['varianza']:.3f}\n"
            f"Desviación estándar: {est['desviacion']:.3f}\n"
            f"Mediana: {est['mediana']:.3f}\n"
            f"MAD: {est['mad']:.3f}  (sigma_MAD = {est['sigma_mad']:.3f})"
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
        descripcion = f"ROI {indice + 1}"

    if excluir_marco:
        region, (df, dc) = quitar_marco(region)
        desplazamiento = (desplazamiento[0] + df, desplazamiento[1] + dc)
        descripcion += " (sin marco)"

    return region, origen + np.asarray(desplazamiento, dtype=float), descripcion


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
            color=PALETA_ROIS[i % len(PALETA_ROIS)], label=f"ROI {i + 1}",
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
        f"impulsos = {100 * ind['fraccion_impulsos']:.2f} %\n"
        f"pico FFT = {ind['pico_fft']:.2f}\n"
        f"fondo borde/centro = {ind['relacion_borde_centro']:.2f}\n"
        f"E altas / E medias = {ind['relacion_altas']:.3f}\n\n"
        f"Hipótesis sugerida:\n{res['hipotesis']}"
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
    """Muestra una figura de matplotlib en una ventana Qt aparte."""
    ventana = QWidget()
    ventana.setWindowTitle(titulo)
    layout = QVBoxLayout(ventana)
    canvas = FigureCanvasQTAgg(fig)
    layout.addWidget(NavigationToolbar2QT(canvas, ventana))
    layout.addWidget(canvas)
    ventana.resize(1500, 1100)
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

        # La hipótesis sugerida precarga el registro del pipeline
        if res["hipotesis"] in widget_pipeline.diagnostico.choices:
            widget_pipeline.diagnostico.value = res["hipotesis"]
        widget_pipeline.observaciones.value = (
            f"Act1 {descripcion}: " + " | ".join(res["evidencias"])
        )

        est = res["estadisticas"]
        informe.value = (
            "INFORME - DIAGNÓSTICO\n"
            f"Capa: {imagen.name}\n"
            f"Región: {descripcion} ({region.shape[0]}x{region.shape[1]})\n"
            f"Media: {est['media']:.2f}   Varianza: {est['varianza']:.2f}\n"
            f"Desvío: {est['desviacion']:.2f}   MAD: {est['mad']:.2f}\n"
            f"σ wavelet: {res['wavelet']['sigma_donoho']:.2f}   "
            f"σ dif (MAD): {res['diferencias']['sigma_mad']:.2f}\n"
            f"\nHIPÓTESIS SUGERIDA: {res['hipotesis']}\n"
            + "\n".join(f"- {e}" for e in res["evidencias"])
            + ("\n\nFigura: " + str(ruta_figura) if ruta_figura else "")
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo diagnosticar:\n{error}"


# ============================================================
# 7. REGISTRO DEL PIPELINE (Actividad 7 - documentación)
#    Carga un paso (diagnóstico/método/parámetros/métricas) al CSV.
#    Los campos los completa el usuario; el CSV se guarda con el botón.
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
    global registros_pendientes

    if imagen is None:
        informe.value = "INFORME\nSeleccioná una capa Image."
        return

    registro = crear_registro(
        imagen=imagen.name,
        diagnostico=diagnostico,
        metodo=metodo,
        parametros=parametros,
        estimacion_ruido=estimacion_ruido,
        metrica_antes=metrica_antes,
        metrica_despues=metrica_despues,
        observaciones=observaciones,
    )
    registros_pendientes = [registro]

    informe.value = (
        "INFORME\n"
        f"Paso preparado para: {imagen.name}\n"
        f"Diagnóstico: {diagnostico}\n"
        f"Método: {metodo or '(vacío)'}"
    )
    estado_csv.value = "CSV\n1 resultado pendiente de guardado."


# ============================================================
# 8. CONTROLES DEL CSV
# ============================================================

ruta_csv = LineEdit(
    value="resultados/resultados_tp1.csv",
    label="Archivo CSV",
)

boton_guardar_csv = PushButton(text="Guardar en CSV")


@boton_guardar_csv.clicked.connect
def guardar_resultados_actuales():
    global registros_pendientes
    try:
        cantidad = len(registros_pendientes)
        ruta = guardar_registros_csv(ruta_csv.value, registros_pendientes)
        registros_pendientes = []
        estado_csv.value = (
            "CSV\n"
            f"{cantidad} fila(s) agregada(s).\n"
            f"Archivo: {ruta}"
        )
    except Exception as error:
        estado_csv.value = f"CSV\nNo se pudo guardar:\n{error}"


# ============================================================
# 9. PANEL LATERAL
# ============================================================

panel = Container(
    widgets=[
        boton_crear_roi,
        widget_gris,
        widget_medicion,
        boton_crear_perfil,
        widget_diagnostico,
        widget_pipeline,
        informe,
        ruta_csv,
        boton_guardar_csv,
        estado_csv,
    ]
)

viewer.window.add_dock_widget(
    panel,
    area="right",
    name="TP1 - Preprocesamiento",
)


# ============================================================
# 10. INICIAR NAPARI
# ============================================================

if __name__ == "__main__":
    napari.run()

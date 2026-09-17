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
# ESTADO: ESQUELETO INICIAL.
#
# Lo que YA funciona (base de la clase, para arrancar):
#   - abrir napari con el panel del TP;
#   - crear una capa de ROIs y medir media/varianza/desv (Act. 1 y 2);
#   - registrar cada paso del pipeline y exportar el CSV acumulativo (Act. 7).
#
# Lo que hay que COMPLETAR (marcado con  # >>> ACÁ COMPLETAMOS ):
#   - Actividad 1: diagnóstico avanzado (FFT, autocorrelación, wavelet, mapas...)
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

from magicgui import magicgui
from magicgui.widgets import (
    Container,
    Label,
    PushButton,
    LineEdit,
)
from napari.layers import Image, Shapes
from skimage.draw import polygon


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


def obtener_datos_roi(imagen_np, roi_layer, indice):
    """
    Devuelve los píxeles de la imagen contenidos dentro de una ROI
    (una figura de la capa Shapes).
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
def diagnosticar(imagen, **kwargs):
    # >>> ACÁ COMPLETAMOS: análisis para formular la hipótesis de degradación
    raise NotImplementedError("Actividad 1: diagnóstico todavía sin implementar.")


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
    widget_medicion.roi.value = roi_layer
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
        img = np.asarray(imagen.data)
        if img.ndim != 2:
            raise ValueError(
                "Para esta práctica se utilizarán imágenes 2D en escala de grises."
            )

        if modo == "Imagen completa":
            datos = img
            region = "Imagen completa"
        else:
            if roi is None:
                raise ValueError("Creá primero una capa de ROIs.")
            indice = obtener_indice_roi_seleccionada(roi)
            datos = obtener_datos_roi(img, roi, indice)
            region = f"ROI {indice + 1} seleccionada"

        n, media, varianza, desviacion = calcular_estadisticas(datos)

        informe.value = (
            "INFORME\n"
            f"Capa: {imagen.name}\n"
            f"Región: {region}\n"
            f"Número de píxeles: {n}\n"
            f"Media: {media:.3f}\n"
            f"Varianza muestral: {varianza:.3f}\n"
            f"Desviación estándar: {desviacion:.3f}"
        )
    except Exception as error:
        informe.value = f"INFORME\nNo se pudo calcular:\n{error}"


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
        widget_medicion,
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

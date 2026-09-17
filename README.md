# PAIB_TP1_AlvarezTaboada_Cobian

**TP1 – PAIByB · Preprocesamiento y evaluación de calidad de bioimágenes**

Widget de [napari](https://napari.org) para diagnosticar, preprocesar y evaluar
bioimágenes afectadas por distintas degradaciones (ruido, fondo no uniforme,
artefactos periódicos, pérdida de resolución). El flujo de trabajo es:

> diagnosticar → estimar → seleccionar → procesar → evaluar

No se aplican todos los métodos a todas las imágenes: para cada una se decide
qué técnica usar, cómo parametrizarla y cómo verificar si hubo mejora real.

**Autores:** Alvarez Taboada · Cobian

---

## Estructura del repo

| Archivo / carpeta            | Qué es                                                            |
| ---------------------------- | ---------------------------------------------------------------- |
| `widget_tp1.py`              | Widget principal de napari (acá vamos completando las actividades) |
| `pixi_multiplataforma.toml`  | Definición del entorno (Python, napari, plugins). **No borrar.**  |
| `data/`                      | Imágenes de retina provistas para el TP                           |
| `resultados/`                | CSV acumulativo e imágenes procesadas (salidas)                  |
| `.gitignore`                 | Ignora el entorno local (`.pixi/`, `pixi.toml`, `pixi.lock`)     |

> El `pixi.toml`, el `pixi.lock` y la carpeta `.pixi/` se **generan localmente**
> y no se versionan: cada uno los reconstruye desde `pixi_multiplataforma.toml`.

---

## 1. Instalar Pixi (una sola vez)

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

**macOS / Linux:**

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Cerrar y volver a abrir la terminal, y verificar:

```bash
pixi --version
```

## 2. Crear el entorno

Desde la carpeta del proyecto.

**Windows (PowerShell):**

```powershell
Copy-Item pixi_multiplataforma.toml pixi.toml -Force
pixi install
pixi run check
pixi run check-ft
```

**macOS / Linux:**

```bash
cp pixi_multiplataforma.toml pixi.toml
pixi install
pixi run check
pixi run check-ft
```

Si algo falla, se puede reconstruir todo borrando `pixi.lock` y `.pixi/` y
volviendo a copiar el `.toml` + `pixi install`.

## 3. Correr el widget

```bash
pixi run python widget_tp1.py
```

Para probar solo que napari abre:

```bash
pixi run napari
```

Dentro de napari: `File → Open File(s)…` para cargar una imagen de `data/`,
y usar el panel **TP1 – Preprocesamiento** de la derecha.

---

## Flujo de trabajo con Git

**Cada vez que trabajan (los dos):**

```bash
git pull            # SIEMPRE antes de empezar
# ...trabajan...
git add .
git commit -m "qué hiciste"
git push
```

Regla de oro: `git pull` antes de `git push`, y en lo posible no editar el
mismo archivo al mismo tiempo para evitar conflictos.

---

## Actividades (checklist del TP)

- [ ] **1.** Diagnóstico (histograma, perfiles, MAD, mapas de variabilidad, FFT, autocorrelación, wavelet)
- [ ] **2.** Estimación de ruido
- [ ] **3.** Reducción de ruido (gaussiano, bilateral, difusión anisotrópica, NLM, TV, wavelet, Wiener local, BM3D)
- [ ] **4.** Restauración (deconvolución Wiener / Richardson–Lucy) y filtros notch para artefactos periódicos
- [ ] **5.** Corrección de fondo (dark/flat-field, rolling ball, white top-hat, homomórfica)
- [ ] **6.** Realce (gamma, log, sigmoidal, CLAHE, pasa-altos/banda, unsharp)
- [ ] **7.** Evaluación con y sin referencia + CSV acumulativo

## Entregables 

- [ ] **1.** Widget funcional y entorno reproducible
- [ ] **2.** Código organizado en módulos
- [ ] **3.** CSV de resultados
- [ ] **4.** Imágenes obtenidas
- [ ] **5.** Informe breve y descriptivo del TP

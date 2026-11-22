# Trabajo 1 — Optimización numérica, metaheurística y combinatoria

Este repositorio resuelve los dos problemas del trabajo. Cada uno se ejecuta **con un solo comando** desde la terminal; los experimentos del punto 2.2 se ejecutan, además, desde un **cuaderno de Jupyter** en el que se cambian los hiperparámetros:

| Problema | Qué resuelve | Carpeta | Tiempo |
|---|---|---|---|
| **1** | Buscar el mínimo de las funciones Rosenbrock y Rastrigin (2D y 3D) con descenso por gradiente, algoritmo genético, PSO y evolución diferencial | `punto_1/` | < 1 min |
| **2.1** | Ruta más **corta** para recorrer las 47 capitales de la España peninsular (TSP) con colonia de hormigas (ACO) y algoritmo genético (GA) | `punto_2/parte_1/` | ~10 s |
| **2.2** | Ruta más **barata** para el mismo recorrido: tiempo + combustible + peajes reales de 2026, y cómo cambia con el valor de la hora | `punto_2/parte_2/` (cuaderno `Experimentos_parte_2.ipynb`) | ~1 min el cuaderno; ~5 min el barrido de sensibilidad |

Todos los datos están dentro del repositorio: **no hace falta conexión a internet** para ejecutar nada.

---

## 1. Instalación (una sola vez)

**Requisito:** Python **3.10 a 3.13**. Compruébalo con `python3 --version`. Si tu `python3` es 3.14 o posterior, crea el entorno con una versión compatible (por ejemplo, `python3.12 -m venv .venv`).

```bash
git clone https://github.com/MgtDevNet/T01_REDES_NEURONALES_CORRECCION_1.git
cd T01_REDES_NEURONALES_CORRECCION_1

python3 -m venv .venv                 # crea un entorno aislado
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt       # instala las versiones exactas
```

> En Windows, escribe `python` en lugar de `python3` en todos los comandos.

Las versiones de las librerías están fijadas en `requirements.txt`, y todos los programas usan **semillas fijas**: la misma configuración da siempre el mismo resultado.

Cada vez que abras una terminal nueva, activa el entorno antes de ejecutar: `source .venv/bin/activate`.

---

## 2. Problema 1 — Rosenbrock y Rastrigin

```bash
cd punto_1
python3 funcion1.py check                               # verificación rápida (1 s)
python3 funcion1.py run   --algo pso --dim 2 --seed 42  # una corrida
python3 funcion1.py stats --runs 30 --dims 2 3          # comparación de algoritmos (~45 s)
python3 funcion1.py gif   --algo gradient_descent       # animación 2D (~10 s)
```

| Comando | Qué hace | Dónde guarda |
|---|---|---|
| `check` | Comprueba que todo funciona | — |
| `run` | Una corrida de un algoritmo | pantalla |
| `stats` | 30 corridas por algoritmo: media, desviación, % de éxito | `results/stats_rosenbrock.csv` |
| `gif` | Animación de la búsqueda (solo 2D) | `figs/` |

**Opciones útiles**
- Algoritmo: `--algo gradient_descent | genetic | pso | differential_evolution`
- Función: `--problem rosenbrock | rastrigin`
- Hiperparámetros sin tocar el código: `--param pso.inercia=0.6` o `--config config.json`

Todos los algoritmos tienen el **mismo presupuesto de evaluaciones** de la función (`--max-evals`, por defecto 20 000), para que la comparación sea justa.

El cuaderno `Experimentos.ipynb` contiene los experimentos del informe.

---

## 3. Problema 2.1 — Ruta más corta (TSP)

```bash
cd punto_2/parte_1
python3 tsp_bio.py
```

**Qué verás:** la ruta tramo a tramo de cada algoritmo y una tabla comparativa con la distancia a la solución óptima exacta (6207,7 km).

**Resultados:** `resultados_puro.csv` (orden de las ciudades) y `resultados_puro.png` (mapa y convergencia).

**Parámetros:** se cambian en `config.ini` (ciudad de inicio, semilla, hormigas, generaciones…). Opciones útiles:
```bash
python3 tsp_bio.py --modo hibrido     # añade búsqueda local (2-opt + Or-opt)
python3 tsp_bio.py --al-reves         # compara la ruta en los dos sentidos
python3 optimo_exacto.py              # óptimo exacto con programación lineal entera (~10 s)
```

Las distancias son kilómetros **por carretera** (OSRM sobre OpenStreetMap), y la ida puede medir distinto que la vuelta.

---

## 4. Problema 2.2 — Ruta más barata (tiempo + combustible + peajes)

**Experimentos: el cuaderno `punto_2/parte_2/Experimentos_parte_2.ipynb`.** Abre el cuaderno
(`jupyter notebook punto_2/parte_2/Experimentos_parte_2.ipynb`, o desde VS Code), cambia los
hiperparámetros en la celda de la sección 1 y ejecuta todas las celdas (*Run All*, ~1 min). El cuaderno
solo ejecuta los dos experimentos, el híbrido (con búsqueda local) y el puro, y muestra la ruta, el mapa
y el GIF de cada algoritmo. La explicación de los datos y del modelo de costo está en el informe
(`docs/index.html`), y el estudio de sensibilidad al valor de la hora se ejecuta desde la terminal con
`python3 src/sensibilidad.py` (ver más abajo).

> Al ejecutarse, el cuaderno sobrescribe `results/hibrido/` y `results/puro/`, que usa el informe. Si
> cambias hiperparámetros para probar, restaura los resultados con `git checkout -- punto_2/parte_2/results`.

> **El cuaderno debe usar el Python del entorno `.venv`** (sección 1):
> - **Jupyter en el navegador:** activa el entorno (`source .venv/bin/activate`) antes de ejecutar `jupyter notebook`.
> - **VS Code:** abre el cuaderno, pulsa *Select Kernel* (arriba a la derecha) → *Python Environments* y elige `.venv/bin/python` (en Windows, `.venv\Scripts\python.exe`). Si VS Code pide instalar `ipykernel`, acepta.
>
> Si la primera celda da `ModuleNotFoundError: No module named 'numpy'`, el cuaderno está usando otro Python: cambia el kernel.

También se puede ejecutar desde la terminal, con los parámetros de `config.ini`:

```bash
cd punto_2/parte_2
python3 src/tsp_espana.py                     # ACO y GA con búsqueda local + óptimo exacto + GIF (~30 s)
python3 src/tsp_espana.py --modo puro         # ACO y GA clásicos: GIF de la evolución (~30 s)
python3 src/sensibilidad.py                   # ¿cambia la ruta al variar el valor de la hora? (~4-5 min)
python3 src/sensibilidad.py --vh-max 40       # solo el rango 0-40 €/h (más rápido)
```

**Modelo de costo.** Para cada par de capitales hay dos rutas reales por carretera, la más rápida
(que puede usar autopistas de pago) y la más rápida **sin peajes**. El vendedor elige la más barata:

```
costo(i→j) = min( con peaje , sin peaje )  de
             valor_hora × horas  +  km × 4,7 L/100 km × 1,799 €/L  +  peaje
             horas, km : OpenRouteService sobre OpenStreetMap (noviembre de 2026)
             peaje     : km en cada autopista × su tarifa de 2026 (data/tarifas_peaje_2026.csv)
```

Como pagar el peaje ahorra tiempo pero cuesta dinero, **la ruta óptima depende del valor de la hora**.

**Qué verás:** la ruta de cada algoritmo tramo a tramo (km, horas, peaje y autopistas usadas), la
comparación ACO vs GA y la distancia (gap) al **óptimo exacto**, calculado con programación lineal entera.

**Resultados:**

| Archivo | Contenido |
|---|---|
| `results/<modo>/ruta_aco.csv`, `ruta_ga.csv` | Orden de las ciudades; en cada tramo, variante elegida (peaje/libre) y costos |
| `results/<modo>/evolucion_*.gif` | Evolución de la mejor solución sobre el mapa de España |
| `results/<modo>/convergencia_*.png` | Mejor costo por iteración, con el óptimo exacto como referencia |
| `results/<modo>/matriz_*.csv` | Matriz de costos y sus componentes (km, horas, combustible, peajes) |
| `results/<modo>/resumen.csv` | Totales de cada algoritmo |
| `results/sensibilidad/` | Barrido del valor de la hora: umbrales, figuras y rutas distintas |

`<modo>` es `hibrido` (por defecto) o `puro`.

**Parámetros:** se cambian en `config.ini` (valor de la hora, combustible, consumo, modo, parámetros
de ACO y GA, rango del barrido). También desde la terminal:
```bash
python3 src/tsp_espana.py --valor-hora 60 --algorithm ga --seed 7 --sin-gif
python3 src/tsp_espana.py --help
```

**Datos y fuentes:** ver `data/README_DATOS.md`. Las rutas se obtuvieron con un servidor
OpenRouteService local (`recoleccion_datos/`). **No hace falta ejecutarlo**: los datos ya están en
`data/`. Para regenerarlos:
```bash
cd punto_2/parte_2/recoleccion_datos
mkdir -p ors-files && wget -O ors-files/spain-latest.osm.pbf https://download.geofabrik.de/europe/spain-latest.osm.pbf
REBUILD_GRAPHS=true docker compose up -d      # construye el grafo (~30 min la primera vez)
pip install -r requirements-recoleccion.txt
python3 vias_peaje_osm.py                     # inventario de vías de peaje de OSM (~13 min)
python3 rutas_con_sin_peaje.py                # 4324 rutas con y sin peaje (~2 min)
```

---

## 5. Estructura del repositorio

```
├── README.md                 ← este archivo
├── requirements.txt          ← versiones exactas de las librerías
├── punto_1/
│   ├── funcion1.py           ← problema 1 (todos los algoritmos)
│   └── Experimentos.ipynb
└── punto_2/
    ├── parte_1/              ← ruta más corta
    │   ├── tsp_bio.py        ← programa principal
    │   ├── config.ini        ← parámetros
    │   ├── datos_tsp.py      ← carga de datos
    │   ├── optimo_exacto.py  ← óptimo exacto (verificación)
    │   └── *.csv             ← coordenadas y distancias
    └── parte_2/              ← ruta más barata
        ├── Experimentos_parte_2.ipynb ← cuaderno: hiperparámetros y experimentos 1 y 2
        ├── src/tsp_espana.py ← programa principal (usa ACO y GA de parte_1)
        ├── src/sensibilidad.py ← barrido del valor de la hora
        ├── config.ini        ← parámetros
        ├── data/             ← rutas con/sin peaje, tarifas, fuentes (README_DATOS.md)
        ├── results/          ← se regenera al ejecutar
        └── recoleccion_datos/← servidor ORS y scripts de obtención de datos (opcional)
```

---

## 6. Problemas frecuentes

| Problema | Solución |
|---|---|
| `python3: command not found` (Windows) | Usa `python` |
| `ModuleNotFoundError: No module named 'numpy'` | Activa el entorno (`source .venv/bin/activate`) y repite `pip install -r requirements.txt` |
| Ese mismo error, pero dentro del cuaderno | El cuaderno usa otro Python: en VS Code, *Select Kernel* → `.venv/bin/python` (ver sección 4) |
| `No se encontró config.ini` | Ejecuta el comando desde la carpeta indicada en cada sección |
| `Permission denied` en `recoleccion_datos/` | Son archivos creados por Docker: `sudo chown -R $USER:$USER punto_2/parte_2/recoleccion_datos` |

#!/usr/bin/env python3
"""TSP de las 47 capitales con costos reales (datos de noviembre de 2026): ACO y GA.

Costo de ir de la capital i a la j, con el valor de la hora Vh (€/h):

    para cada variante r ∈ {con_peaje, sin_peaje} de la ruta por carretera:
        C_ij^r(Vh) = Vh · horas_ij^r  +  k · km_ij^r  +  peaje_ij^r
    C_ij(Vh) = min(C_ij^con_peaje, C_ij^sin_peaje)

    k = consumo/100 · precio del combustible  (€/km)
    horas, km : OpenRouteService (servidor local, mapa OSM del 17-nov-2026)
    peaje     : Σ km recorridos en cada autopista · tarifa €/km de 2026 (data/tarifas_peaje_2026.csv)

Para cada tramo el vendedor elige pagar el peaje (ruta más rápida) o ir por la red
libre (más lenta pero gratis): por eso la ruta óptima SÍ depende de Vh.

Los algoritmos son los de la parte 1 (punto_2/parte_1/tsp_bio.py): ACO elitista y GA
(con búsqueda local 2-opt + Or-opt opcional), aplicados a la matriz de costos en €.

Ejecución:   python3 src/tsp_espana.py            (desde punto_2/parte_2)
Parámetros:  config.ini  ·  opciones de terminal (--help)
Prioridad:   terminal  >  config.ini  >  valores por defecto
"""

from pathlib import Path
import argparse
import configparser
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

if "ipykernel" not in sys.modules:                 # en un notebook se usa el backend inline
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]          # .../punto_2/parte_2
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PARTE_1 = ROOT.parent / "parte_1"

sys.path.insert(0, str(PARTE_1))
import tsp_bio as tb                                 # noqa: E402  (ACO, GA, 2-opt, Or-opt)
import optimo_exacto as ox                           # noqa: E402  (óptimo exacto por PLE)

VARIANTES = ("con_peaje", "sin_peaje")


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN (config.ini)
# ═════════════════════════════════════════════════════════════════════════════
# Si una línea falta en config.ini se usa el valor de aquí. El tipo del valor
# por defecto (int, float, bool, str) indica cómo se lee el texto.
POR_DEFECTO = {
    "general": {
        "algoritmo": "both",            # aco | ga | both
        "semilla": 20261008,
        "inicio": "Madrid",             # solo cambia desde dónde se MUESTRA el ciclo
        "generar_gif": True,
        "verificar_exacto": True,       # compara con el óptimo exacto (PLE, ~3 s)
        "modo": "hibrido",              # hibrido = con búsqueda local · puro = ACO/GA clásicos
        "resultados": "results",        # carpeta de salida (relativa a parte_2/); se añade /<modo>
    },
    "costos": {
        "valor_hora": 26.18,            # €/h (INE, coste laboral por hora, 2T 2026)
        "precio_combustible": 1.799,    # €/L (gasolina 95, 5-oct-2026)
        "consumo_l_100km": 4.7,         # Toyota Corolla Hybrid 140H (WLTP, extremo alto)
    },
    "aco": {
        "n_hormigas": 20,
        "iteraciones": 200,
        "alpha": 1.0,                   # peso de la feromona
        "beta": 3.0,                    # peso de la heurística 1/costo
        "rho": 0.1,                     # evaporación
        "elitismo": 5.0,                # refuerzo extra de la mejor ruta global
        "busqueda_local": True,         # 2-opt + Or-opt a la mejor hormiga de cada iteración
    },
    "ga": {
        "tam_poblacion": 150,
        "generaciones": 300,
        "p_cruce": 0.9,
        "p_mutacion": 0.3,
        "p_busqueda_local": 0.05,       # fracción de hijos pulidos con 2-opt + Or-opt
        "n_elite": 2,
        "k_torneo": 3,
    },
    "puro": {                           # sustituye a [aco]/[ga] cuando modo = puro
        "aco_n_hormigas": 47,
        "aco_iteraciones": 300,
        "ga_generaciones": 1500,
    },
    "sensibilidad": {
        "vh_min": 0.0,                  # €/h
        "vh_max": 100.0,
        "vh_paso": 1.0,
    },
}
VERDADERO, FALSO = {"si", "sí", "true", "1", "yes"}, {"no", "false", "0"}


def convertir(texto, ejemplo, donde):
    """Convierte el texto del .ini al tipo del valor por defecto."""
    t = texto.strip()
    try:
        if isinstance(ejemplo, bool):               # bool antes que int (bool es subclase de int)
            if t.lower() in VERDADERO | FALSO:
                return t.lower() in VERDADERO
            raise ValueError
        if isinstance(ejemplo, int):
            return int(t)
        if isinstance(ejemplo, float):
            return float(t.replace(",", "."))       # tolera la coma decimal
        return t
    except ValueError:
        tipo = {bool: "si/no", int: "entero", float: "número"}[type(ejemplo)]
        sys.exit(f"Error en {donde} = '{t}': se esperaba {tipo}.")


def cargar_config(ruta):
    """Lee config.ini sobre los valores por defecto. Avisa de erratas en los nombres."""
    cfg = {sec: dict(vals) for sec, vals in POR_DEFECTO.items()}
    ruta = Path(ruta)
    if not ruta.is_absolute() and not ruta.exists():
        ruta = ROOT / ruta                           # una ruta relativa se busca también en parte_2/
    if not ruta.exists():
        print(f"AVISO: no se encontró {ruta}. Se usan los valores por defecto.")
        return cfg
    ini = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    ini.read(ruta, encoding="utf-8-sig")             # -sig: tolera el BOM del Bloc de notas
    print(f"Configuración leída de: {ruta}")
    for sec in ini.sections():
        if sec not in cfg:
            sys.exit(f"Error en {ruta}: sección desconocida [{sec}].")
        for clave, texto in ini.items(sec):
            if clave not in cfg[sec]:                # detecta erratas: 'betta', 'n_hormiga'...
                sys.exit(f"Error en {ruta} → [{sec}]: parámetro desconocido '{clave}'. "
                         f"Válidos: {', '.join(cfg[sec])}")
            cfg[sec][clave] = convertir(texto, POR_DEFECTO[sec][clave], f"[{sec}] {clave}")
    return cfg


def validar_config(cfg, ciudades):
    g, a, ga_, c, s = cfg["general"], cfg["aco"], cfg["ga"], cfg["costos"], cfg["sensibilidad"]
    err = []
    if g["modo"] not in ("hibrido", "puro"):
        err.append("[general] modo debe ser hibrido o puro")
    if g["algoritmo"] not in ("aco", "ga", "both"):
        err.append("[general] algoritmo debe ser aco, ga o both")
    if g["inicio"] not in ciudades:
        err.append(f"[general] inicio: ciudad desconocida '{g['inicio']}'")
    if a["n_hormigas"] < 1 or a["iteraciones"] < 1:
        err.append("[aco] n_hormigas e iteraciones deben ser ≥ 1")
    if not 0 < a["rho"] <= 1:
        err.append("[aco] rho debe estar en (0, 1]")
    if ga_["tam_poblacion"] < 2 or ga_["generaciones"] < 1:
        err.append("[ga] tam_poblacion ≥ 2 y generaciones ≥ 1")
    for p in ("p_cruce", "p_mutacion", "p_busqueda_local"):
        if not 0 <= ga_[p] <= 1:
            err.append(f"[ga] {p} debe estar entre 0 y 1")
    if not 0 <= ga_["n_elite"] < ga_["tam_poblacion"]:
        err.append("[ga] n_elite debe estar entre 0 y tam_poblacion - 1")
    if not 1 <= ga_["k_torneo"] <= ga_["tam_poblacion"]:
        err.append("[ga] k_torneo debe estar entre 1 y tam_poblacion")
    if c["valor_hora"] < 0 or c["precio_combustible"] < 0 or c["consumo_l_100km"] < 0:
        err.append("[costos] valor_hora, precio_combustible y consumo deben ser ≥ 0")
    if s["vh_paso"] <= 0 or s["vh_max"] < s["vh_min"]:
        err.append("[sensibilidad] vh_paso > 0 y vh_max ≥ vh_min")
    if err:
        sys.exit("Errores en la configuración:\n  - " + "\n  - ".join(err))


# ═════════════════════════════════════════════════════════════════════════════
# DATOS Y MODELO DE COSTO
# ═════════════════════════════════════════════════════════════════════════════
def cargar_datos():
    """Lee coordenadas, rutas con/sin peaje y tarifas. Devuelve un dict de matrices 47×47."""
    for f in ("coordenadas.csv", "rutas_con_peaje.csv", "rutas_sin_peaje.csv",
              "tarifas_peaje_2026.csv"):
        if not (DATA / f).is_file():
            sys.exit(f"Falta el archivo de datos {DATA / f}")
    coords = pd.read_csv(DATA / "coordenadas.csv", encoding="utf-8-sig")
    names = coords["Capital"].astype(str).tolist()
    idx = {c: k for k, c in enumerate(names)}
    n = len(names)
    tarifas = pd.read_csv(DATA / "tarifas_peaje_2026.csv").set_index("via")["eur_km"].to_dict()

    datos = {"names": names, "coords": coords, "tarifas": tarifas}
    for v in VARIANTES:
        df = pd.read_csv(DATA / f"rutas_{v}.csv")
        km, horas, peaje = (np.zeros((n, n)) for _ in range(3))
        vias = np.full((n, n), "", dtype=object)
        for r in df.itertuples():
            i, j = idx[r.origen], idx[r.destino]
            por_via = json.loads(r.peaje_por_via)
            sin_tarifa = set(por_via) - set(tarifas)
            if sin_tarifa:
                sys.exit(f"Vías de peaje sin tarifa en tarifas_peaje_2026.csv: {sorted(sin_tarifa)}")
            km[i, j], horas[i, j] = r.km, r.horas
            peaje[i, j] = sum(x * tarifas[via] for via, x in por_via.items())
            vias[i, j] = ", ".join(f"{via} {x:.0f} km" for via, x in por_via.items() if x >= 1)
        faltan = n * (n - 1) - len(df)
        if faltan:
            sys.exit(f"rutas_{v}.csv incompleto: faltan {faltan} pares")
        datos[v] = {"km": km, "horas": horas, "peaje": peaje, "vias": vias}
    return datos


def matriz_costos(datos, vh, precio, consumo):
    """Costo de cada tramo eligiendo, para cada par, la variante (con/sin peaje) más barata.

    Devuelve un dict con la matriz de costo C y los componentes de la variante elegida."""
    k = consumo / 100 * precio                                   # €/km de combustible
    total = {v: vh * datos[v]["horas"] + k * datos[v]["km"] + datos[v]["peaje"]
             for v in VARIANTES}
    con = total["con_peaje"] <= total["sin_peaje"]               # empate: la más rápida
    elige = lambda campo: np.where(con, datos["con_peaje"][campo], datos["sin_peaje"][campo])
    m = {
        "costo": np.where(con, total["con_peaje"], total["sin_peaje"]),
        "km": elige("km"), "horas": elige("horas"), "peaje": elige("peaje"),
        "vias": elige("vias"), "usa_con_peaje": con,
    }
    m["combustible"] = k * m["km"]
    m["tiempo"] = vh * m["horas"]
    m["paga_peaje"] = m["peaje"] > 0.005
    for M in m.values():
        if M.dtype != object:
            np.fill_diagonal(M, 0)
    return m


def costo_ciclo(tour, C):
    return float(sum(C[tour[i - 1], tour[i]] for i in range(len(tour))))


# ═════════════════════════════════════════════════════════════════════════════
# ALGORITMOS (los de la parte 1 sobre la matriz de costos)
# ═════════════════════════════════════════════════════════════════════════════
def usar_matriz(C):
    """Pone la matriz de costos como matriz activa de tsp_bio (la parte 1 usa una global)."""
    tb.D = C.tolist()
    tb.N = len(C)
    tb.SIMETRICA = bool(np.allclose(C, C.T))


def resolver(alg, C, cfg, semilla, registrar=False):
    """Ejecuta ACO o GA de la parte 1. Devuelve (ruta, costo, historial, it_mejor, mejoras).

    mejoras = [(iteración, ruta, costo), ...] cada vez que mejora la mejor solución
    (solo si registrar=True; se usa para el GIF de la evolución)."""
    usar_matriz(C)
    mejoras = []
    cb = (lambda it, ruta, L: mejoras.append((it, ruta, L))) if registrar else None
    rng = random.Random(semilla)
    if alg == "aco":
        p = cfg["aco"]
        ruta, L, hist, it = tb.aco(rng, n_hormigas=p["n_hormigas"], iteraciones=p["iteraciones"],
                                   alpha=p["alpha"], beta=p["beta"], rho=p["rho"],
                                   elitismo=p["elitismo"], busqueda_local=p["busqueda_local"],
                                   log_cada=0, al_mejorar=cb)
    else:
        p = cfg["ga"]
        ruta, L, hist, it = tb.ga(rng, tam_poblacion=p["tam_poblacion"],
                                  generaciones=p["generaciones"], p_cruce=p["p_cruce"],
                                  p_mutacion=p["p_mutacion"], p_busqueda_local=p["p_busqueda_local"],
                                  n_elite=p["n_elite"], k_torneo=p["k_torneo"],
                                  log_cada=0, al_mejorar=cb)
    return ruta, L, hist, it, mejoras


def optimo_exacto(C):
    """Óptimo global por programación lineal entera (optimo_exacto.py de la parte 1)."""
    L, ruta = ox.resolver(C.tolist())
    return ruta, L


# ═════════════════════════════════════════════════════════════════════════════
# PRESENTACIÓN
# ═════════════════════════════════════════════════════════════════════════════
def rotar(tour, inicio):
    """Mismo ciclo, empezando en la ciudad 'inicio' (el costo no cambia)."""
    k = tour.index(inicio)
    return tour[k:] + tour[:k]


def tabla_ruta(tour, names, m):
    """Una fila por tramo: desde, hasta, variante elegida, km, horas y cada componente del costo."""
    filas, acum = [], 0.0
    for k in range(len(tour)):
        a, b = tour[k], tour[(k + 1) % len(tour)]               # el último tramo regresa al inicio
        acum += m["costo"][a, b]
        filas.append({
            "orden": k + 1, "desde": names[a], "hasta": names[b],
            "variante": "con peaje" if m["paga_peaje"][a, b] else "libre",
            "vias_peaje": m["vias"][a, b],
            "km": m["km"][a, b], "horas": m["horas"][a, b],
            "tiempo_eur": m["tiempo"][a, b], "combustible_eur": m["combustible"][a, b],
            "peaje_eur": m["peaje"][a, b], "costo_eur": m["costo"][a, b],
            "costo_acumulado_eur": acum,
        })
    return pd.DataFrame(filas)


def corto(nombre, w=22):
    """Recorta nombres largos solo para la pantalla (el CSV guarda el nombre completo)."""
    return nombre if len(nombre) <= w else nombre[:w - 1] + "…"


def imprimir_ruta(titulo, df):
    print(f"\n{titulo}")
    print(f"   {'#':>2}  {'Desde':<22}  {'Hasta':<22} {'km':>6} {'horas':>5} {'peaje €':>7} "
          f"{'tramo €':>7} {'acum. €':>8}  vías de peaje")
    for r in df.itertuples():
        print(f"   {r.orden:>2}  {corto(r.desde):<22}  {corto(r.hasta):<22} {r.km:6.1f} "
              f"{r.horas:5.2f} {r.peaje_eur:7.2f} {r.costo_eur:7.2f} {r.costo_acumulado_eur:8.2f}"
              f"  {r.vias_peaje}")


def totales(df):
    return {
        "km": df["km"].sum(), "horas": df["horas"].sum(),
        "tiempo_eur": df["tiempo_eur"].sum(), "combustible_eur": df["combustible_eur"].sum(),
        "peaje_eur": df["peaje_eur"].sum(), "costo_total_eur": df["costo_eur"].sum(),
        "tramos_con_peaje": int((df["variante"] == "con peaje").sum()),
    }


def comparar(res, optimo=None):
    """Tabla ACO vs GA (y frente al óptimo exacto si se calculó)."""
    print("\nCOMPARACIÓN")
    print(f"   {'Algoritmo':<10} {'Costo €':>9} {'Gap %':>6} {'km':>8} {'horas':>6} "
          f"{'Tiempo €':>9} {'Combust. €':>10} {'Peajes €':>9} {'Tramos peaje':>12}")
    for alg, t in res.items():
        gap = f"{100 * (t['costo_total_eur'] / optimo - 1):6.2f}" if optimo else "     -"
        print(f"   {alg:<10} {t['costo_total_eur']:9.2f} {gap} {t['km']:8.1f} {t['horas']:6.1f} "
              f"{t['tiempo_eur']:9.2f} {t['combustible_eur']:10.2f} {t['peaje_eur']:9.2f} "
              f"{t['tramos_con_peaje']:>12}")
    if optimo:
        print(f"   {'Óptimo':<10} {optimo:9.2f}   0.00   (programación lineal entera)")


# ═════════════════════════════════════════════════════════════════════════════
# GIF: evolución de la mejor solución sobre el mapa
# ═════════════════════════════════════════════════════════════════════════════
def dibujar_mapa_base(ax, coords):
    """Contorno de la península (data/contorno_espana.csv, Natural Earth 1:50m) y las capitales.
    España en color crema; Portugal, Francia y Andorra en gris como referencia."""
    contorno = DATA / "contorno_espana.csv"
    ax.set_facecolor("#dfe9f2")                                  # mar
    if contorno.is_file():
        c = pd.read_csv(contorno)
        for (_, pais), g in c.groupby(["parte", "pais"]):
            ax.fill(g["lon"], g["lat"], color="#f3efe6" if pais == "ESP" else "#e2e2e2",
                    ec="#9a9488", lw=0.6, zorder=0)
    ax.scatter(coords["Longitud"], coords["Latitud"], s=12, color="#333", zorder=3)
    ax.set_xlim(-9.6, 3.5)
    ax.set_ylim(35.8, 44.0)
    ax.set_aspect(1 / np.cos(np.radians(40)))
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")


def mapa_ruta(tour, m, names, coords, path, titulo):
    """Mapa estático de una ruta: tramos con peaje en rojo (con la autopista), red libre en azul."""
    lon, lat = coords["Longitud"].to_numpy(), coords["Latitud"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6.6))
    dibujar_mapa_base(ax, coords)
    for k in range(len(tour)):
        a, b = tour[k - 1], tour[k]
        peaje = m["paga_peaje"][a, b]
        ax.annotate("", xy=(lon[b], lat[b]), xytext=(lon[a], lat[a]),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8 if peaje else 1.1, mutation_scale=9,
                                    color="#c0392b" if peaje else "#2c6fb0"), zorder=2)
        if peaje:
            ax.text((lon[a] + lon[b]) / 2, (lat[a] + lat[b]) / 2, m["vias"][a, b].split(" ")[0],
                    fontsize=7, color="#c0392b", ha="center", va="bottom", zorder=4)
    for i in range(len(names)):
        ax.annotate(names[i].split("/")[0], (lon[i], lat[i]), fontsize=5.5, xytext=(2, 2),
                    textcoords="offset points", zorder=4)
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gif_evolucion(mejoras, m, coords, path, titulo, etiqueta="iteración", max_cuadros=45):
    """Un cuadro por cada mejora de la mejor solución (submuestreado si hay muchas).
    Tramos con peaje en rojo, tramos por red libre en azul."""
    if len(mejoras) > max_cuadros:
        sel = np.unique(np.linspace(0, len(mejoras) - 1, max_cuadros).round().astype(int))
        mejoras = [mejoras[i] for i in sel]
    mejoras = mejoras + [mejoras[-1]] * 6                       # pausa al final
    lon, lat = coords["Longitud"].to_numpy(), coords["Latitud"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6.6))
    dibujar_mapa_base(ax, coords)
    lineas = []
    texto = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    ax.set_title(titulo)
    fig.tight_layout()

    def update(f):
        for ln in lineas:
            ln.remove()
        lineas.clear()
        it, ruta, L = mejoras[f]
        for k in range(len(ruta)):
            a, b = ruta[k - 1], ruta[k]
            peaje = m["paga_peaje"][a, b]
            ln, = ax.plot([lon[a], lon[b]], [lat[a], lat[b]], lw=1.6 if peaje else 1.1,
                          color="#c0392b" if peaje else "#2c6fb0", zorder=2)
            lineas.append(ln)
        texto.set_text(f"{etiqueta} {it}  ·  mejor costo {L:,.2f} €\n"
                       f"rojo: tramo con peaje · azul: red libre")
        return lineas + [texto]

    anim = FuncAnimation(fig, update, frames=len(mejoras), interval=350, blit=False)
    anim.save(path, writer=PillowWriter(fps=3))
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# PROGRAMA PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════
def leer_argumentos(descripcion):
    """Opciones de terminal comunes (valen None si no se escriben: así no pisan config.ini)."""
    p = argparse.ArgumentParser(description=descripcion)
    p.add_argument("--config", default=str(ROOT / "config.ini"),
                   help="archivo de parámetros (por defecto: parte_2/config.ini)")
    p.add_argument("--algorithm", choices=["aco", "ga", "both"])
    p.add_argument("--modo", choices=["hibrido", "puro"],
                   help="hibrido = con búsqueda local 2-opt + Or-opt; puro = ACO y GA clásicos")
    p.add_argument("--seed", type=int)
    p.add_argument("--inicio", help="ciudad desde la que se muestra el recorrido")
    p.add_argument("--valor-hora", type=float, help="valor de la hora del vendedor (€/h)")
    p.add_argument("--aco-ants", type=int)
    p.add_argument("--aco-iterations", type=int)
    p.add_argument("--ga-population", type=int)
    p.add_argument("--ga-generations", type=int)
    p.add_argument("--sin-gif", action="store_true", help="no generar los GIF (más rápido)")
    p.add_argument("--sin-exacto", action="store_true", help="no calcular el óptimo exacto")
    return p


def config_desde_argumentos(args):
    cfg = cargar_config(args.config)
    for valor, sec, clave in [
        (args.algorithm, "general", "algoritmo"), (args.seed, "general", "semilla"),
        (args.modo, "general", "modo"),
        (args.inicio, "general", "inicio"), (args.valor_hora, "costos", "valor_hora"),
        (args.aco_ants, "aco", "n_hormigas"), (args.aco_iterations, "aco", "iteraciones"),
        (args.ga_population, "ga", "tam_poblacion"), (args.ga_generations, "ga", "generaciones"),
    ]:
        if valor is not None:                        # la terminal tiene prioridad
            cfg[sec][clave] = valor
    if cfg["general"]["modo"] == "puro":             # ACO y GA clásicos, sin búsqueda local
        aplicar_modo_puro(cfg)
        for valor, sec, clave in [(args.aco_ants, "aco", "n_hormigas"),
                                  (args.aco_iterations, "aco", "iteraciones"),
                                  (args.ga_generations, "ga", "generaciones")]:
            if valor is not None:                    # la terminal sigue mandando
                cfg[sec][clave] = valor
    if args.sin_gif:
        cfg["general"]["generar_gif"] = False
    if args.sin_exacto:
        cfg["general"]["verificar_exacto"] = False
    return cfg


def aplicar_modo_puro(cfg):
    """Con modo = puro, sustituye los parámetros de [aco] y [ga] por los de la sección [puro]."""
    if cfg["general"]["modo"] == "puro":
        pu = cfg["puro"]
        cfg["aco"].update(n_hormigas=pu["aco_n_hormigas"], iteraciones=pu["aco_iteraciones"],
                          busqueda_local=False)
        cfg["ga"].update(generaciones=pu["ga_generaciones"], p_busqueda_local=0.0)
    return cfg


def grafico_convergencia(hist, alg, vh, optimo=None, ax=None):
    """Mejor costo por iteración (ACO) o generación (GA), con el óptimo exacto como referencia."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(hist) + 1), hist, color="#2c6fb0", label=alg.upper())
    if optimo:
        ax.axhline(optimo, ls="--", color="#555", lw=1, label=f"óptimo exacto {optimo:,.2f} €")
    ax.legend()
    ax.set_xlabel("Iteración" if alg == "aco" else "Generación")
    ax.set_ylabel("Mejor costo (€)")
    ax.set_title(f"Convergencia {alg.upper()} (Vh = {vh} €/h)")
    ax.grid(alpha=0.25)
    if fig is not None:
        fig.tight_layout()
    return fig


def ejecutar(cfg, datos=None, verbose=True):
    """Experimento completo con la configuración cfg (la de config.ini o la del notebook).

    Construye la matriz de costos, calcula el óptimo exacto (si se pide), ejecuta ACO y/o GA,
    guarda tablas, figuras y GIF en results/<modo>/ y devuelve un dict con todo:
        m, optimo, ruta_optima, carpeta, resumen (DataFrame) y
        resultados[alg] = {ruta, costo, historial, it_mejor, segundos, tabla, totales, mejoras}
    """
    gen, cst = cfg["general"], cfg["costos"]
    datos = datos or cargar_datos()
    names, coords = datos["names"], datos["coords"]
    validar_config(cfg, names)
    out = ROOT / gen["resultados"] / gen["modo"]                 # results/hibrido o results/puro
    out.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)

    vh, precio, cons = cst["valor_hora"], cst["precio_combustible"], cst["consumo_l_100km"]
    m = matriz_costos(datos, vh, precio, cons)
    C = m["costo"]
    say(f"=== TSP ESPAÑA CON COSTOS: EJECUCIÓN (modo {gen['modo']}) ===")
    say(f"Capitales: {len(names)} | semilla: {gen['semilla']} | valor hora: {vh} €/h | "
        f"combustible: {precio} €/L · {cons} L/100 km ({cons / 100 * precio:.4f} €/km)")
    n_con = int((m["paga_peaje"] & ~np.eye(len(C), dtype=bool)).sum())
    say(f"Tramos (pares ordenados) en los que conviene pagar peaje con Vh = {vh}: "
        f"{n_con} de {len(C) * (len(C) - 1)}")

    for fn, M in [("matriz_costos_eur.csv", C), ("matriz_tiempos_h.csv", m["horas"]),
                  ("matriz_km.csv", m["km"]), ("matriz_combustible_eur.csv", m["combustible"]),
                  ("matriz_peajes_eur.csv", m["peaje"]),
                  ("matriz_usa_peaje.csv", m["paga_peaje"].astype(int))]:
        pd.DataFrame(M, index=names, columns=names).round(4).to_csv(out / fn,
                                                                    index_label="Capital")

    optimo, ruta_opt = None, None
    if gen["verificar_exacto"]:
        t0 = time.time()
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):          # silencia los subtours
            ruta_opt, optimo = optimo_exacto(C)
        ruta_opt = rotar(ruta_opt, names.index(gen["inicio"]))
        say(f"Óptimo exacto (programación lineal entera): {optimo:.2f} € ({time.time() - t0:.0f} s)")

    algos = ["aco", "ga"] if gen["algoritmo"] == "both" else [gen["algoritmo"]]
    res, tot_por_alg, resumen = {}, {}, []
    for alg in algos:
        t0 = time.time()
        tour, val, hist, it_mejor, mejoras = resolver(alg, C, cfg, gen["semilla"],
                                                      registrar=gen["generar_gif"])
        seg = time.time() - t0
        tour = rotar(tour, names.index(gen["inicio"]))
        df = tabla_ruta(tour, names, m)
        if verbose:
            imprimir_ruta(f"RUTA {alg.upper()} — {val:.2f} € (mejor en la iteración {it_mejor}, "
                          f"{seg:.1f} s)", df)
        df.round(4).to_csv(out / f"ruta_{alg}.csv", index=False)
        tot = totales(df)
        tot_por_alg[alg.upper()] = tot

        pd.DataFrame({"iteracion": range(1, len(hist) + 1), "mejor_costo_eur": hist}).to_csv(
            out / f"convergencia_{alg}.csv", index=False)
        fig = grafico_convergencia(hist, alg, vh, optimo)
        fig.savefig(out / f"convergencia_{alg}.png", dpi=140)
        plt.close(fig)
        mapa_ruta(tour, m, names, coords, out / f"mapa_ruta_{alg}.png",
                  f"Ruta {alg.upper()} {gen['modo']} — {val:,.2f} € (Vh = {vh} €/h)")
        if gen["generar_gif"]:
            gif_evolucion(mejoras, m, coords, out / f"evolucion_{alg}.gif",
                          f"Evolución de la mejor ruta — {alg.upper()} {gen['modo']} (Vh = {vh} €/h)",
                          etiqueta="iteración" if alg == "aco" else "generación")
        resumen.append({
            "algoritmo": alg.upper(), "semilla": gen["semilla"], "valor_hora_eur": vh,
            "combustible_eur_l": precio, "consumo_l_100km": cons,
            **{k: round(v, 3) for k, v in tot.items()},
            "gap_optimo_pct": round(100 * (tot["costo_total_eur"] / optimo - 1), 3) if optimo else "",
            "iteracion_mejor": it_mejor, "segundos": round(seg, 1),
            "orden_ciudades": " → ".join([names[i] for i in tour] + [names[tour[0]]]),
        })
        res[alg] = {"ruta": tour, "costo": val, "historial": hist, "it_mejor": it_mejor,
                    "segundos": seg, "tabla": df, "totales": tot, "mejoras": mejoras}
    if verbose:
        comparar(tot_por_alg, optimo)

    resumen = pd.DataFrame(resumen)
    resumen.to_csv(out / "resumen.csv", index=False)
    say(f"\nResultados en {out}")
    return {"m": m, "optimo": optimo, "ruta_optima": ruta_opt, "carpeta": out,
            "resumen": resumen, "resultados": res, "datos": datos}


def main():
    args = leer_argumentos("TSP 47 capitales con costos reales (ACO y GA).").parse_args()
    ejecutar(config_desde_argumentos(args))


if __name__ == "__main__":
    main()

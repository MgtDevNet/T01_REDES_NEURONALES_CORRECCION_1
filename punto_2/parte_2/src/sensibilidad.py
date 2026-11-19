#!/usr/bin/env python3
"""Sensibilidad de la ruta óptima al valor de la hora del vendedor (Vh).

Preguntas del enunciado: ¿cambia la ruta óptima al variar Vh? ¿a partir de qué valor?

Método
  1. Barrido: para cada Vh de la rejilla [vh_min, vh_max] (paso vh_paso, en config.ini)
     se construye la matriz de costos C(Vh) y se resuelve el ÓPTIMO EXACTO (programación
     lineal entera, optimo_exacto.py de la parte 1). Usar el óptimo exacto, y no una
     metaheurística, garantiza que un cambio de ruta no es ruido del algoritmo.
  2. Umbrales: entre dos valores vecinos con rutas distintas, el valor exacto del cambio
     se localiza por bisección (precisión --tolerancia, 0.01 €/h por defecto).
  3. Se distinguen tres tipos de cambio:
       - ORDEN:    cambian las conexiones entre capitales (el ciclo, sin tener en cuenta
                   el sentido).
       - SENTIDO:  el mismo ciclo, recorrido al revés. Ocurre porque los costos son
                   asimétricos (ida ≠ vuelta): al cambiar Vh puede convenir el otro sentido.
       - VARIANTE: algún tramo del ciclo pasa de la red libre a la autopista de peaje
                   (o al revés).
  4. Umbral de cada tramo de la ruta base: con el orden fijo, el tramo i→j pasa a la
     variante con peaje cuando Vh supera
            Vh*_ij = (peaje_con - peaje_sin + k·(km_con - km_sin)) / (horas_sin - horas_con)

Ejecución (desde punto_2/parte_2):
    python3 src/sensibilidad.py                 # rejilla de config.ini (0–100 €/h, paso 1)
    python3 src/sensibilidad.py --vh-max 40     # solo el rango 0–40 €/h (más rápido)

Un paso mayor que 1 NO es más rápido: cada intervalo contiene más cambios y la bisección
los busca uno tras otro; además, un cambio que se deshace dentro de un intervalo se pierde.
Salidas en results/sensibilidad/.
"""

import contextlib
import io
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

if "ipykernel" not in sys.modules:                 # en un notebook se usa el backend inline
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsp_espana as te                              # noqa: E402

SALIDA = te.RESULTS / "sensibilidad"
# Procesos nuevos con "spawn" y no con "fork": copiar con fork un kernel de Jupyter que ya
# tiene hilos de numpy/scipy deja a los hijos bloqueados. Spawn funciona igual en Linux,
# macOS y Windows, en la terminal y en el cuaderno.
_CONTEXTO = multiprocessing.get_context("spawn")
_DATOS = None                                        # datos cargados una vez por proceso


def _datos():
    global _DATOS
    if _DATOS is None:
        _DATOS = te.cargar_datos()
    return _DATOS


def resolver_vh(vh, precio, consumo):
    """Óptimo exacto para un valor de la hora. Devuelve un dict con la ruta y sus totales."""
    d = _datos()
    m = te.matriz_costos(d, vh, precio, consumo)
    with contextlib.redirect_stdout(io.StringIO()):  # silencia los mensajes de subtours
        ruta, L = te.optimo_exacto(m["costo"])
    ruta = te.rotar(ruta, d["names"].index("Madrid"))
    arcos = [(ruta[k - 1], ruta[k]) for k in range(len(ruta))]
    peaje_arcos = frozenset(a for a in arcos if m["paga_peaje"][a])
    suma = lambda M: float(sum(M[a] for a in arcos))
    return {
        "vh": vh, "costo_eur": L, "km": suma(m["km"]), "horas": suma(m["horas"]),
        "tiempo_eur": suma(m["tiempo"]), "combustible_eur": suma(m["combustible"]),
        "peaje_eur": suma(m["peaje"]), "tramos_con_peaje": len(peaje_arcos),
        "ruta": tuple(ruta), "arcos": frozenset(arcos),
        "orden": frozenset(frozenset(a) for a in arcos),      # ciclo sin sentido
        "peaje_arcos": peaje_arcos,
        "peaje_tramos": frozenset(frozenset(a) for a in peaje_arcos),   # sin sentido
    }


TIPOS = ("orden", "sentido", "variante")


def misma(a, b, tipo):
    if tipo == "orden":
        return a["orden"] == b["orden"]
    if tipo == "sentido":                 # solo tiene sentido comparar si el ciclo es el mismo
        return a["orden"] != b["orden"] or a["arcos"] == b["arcos"]
    return a["peaje_tramos"] == b["peaje_tramos"]


def bisecar(r_lo, r_hi, tipo, precio, consumo, tol):
    """Valor de Vh (± tol/2) en el que la solución pasa de ser como r_lo a ser como r_hi."""
    lo, hi = r_lo, r_hi
    while hi["vh"] - lo["vh"] > tol:
        mid = resolver_vh((lo["vh"] + hi["vh"]) / 2, precio, consumo)
        if misma(mid, lo, tipo):
            lo = mid
        else:
            hi = mid
    return lo, hi


def buscar_cambios(r_a, r_b, tipo, precio, consumo, tol):
    """Todos los cambios de un tipo entre dos valores de la rejilla (puede haber más de uno)."""
    lo, hi = bisecar(r_a, r_b, tipo, precio, consumo, tol)
    cambios = [(lo, hi)]
    if not misma(hi, r_b, tipo):              # queda otro cambio entre hi y r_b
        cambios += buscar_cambios(hi, r_b, tipo, precio, consumo, tol)
    return cambios


def describir(lo, hi, tipo, names):
    flecha = lambda a: f"{names[a[0]]}→{names[a[1]]}"
    if tipo == "orden":
        sale = sorted(a for a in lo["arcos"] if frozenset(a) not in hi["orden"])
        entra = sorted(a for a in hi["arcos"] if frozenset(a) not in lo["orden"])
        return ("salen: " + "; ".join(map(flecha, sale)) + " | entran: " + "; ".join(map(flecha, entra)))
    if tipo == "sentido":
        return "el mismo ciclo pasa a recorrerse en sentido contrario"
    pasa = sorted(a for a in hi["peaje_arcos"] if frozenset(a) not in lo["peaje_tramos"])
    deja = sorted(a for a in lo["peaje_arcos"] if frozenset(a) not in hi["peaje_tramos"])
    txt = []
    if pasa:
        txt.append("pasan a peaje: " + "; ".join(map(flecha, pasa)))
    if deja:
        txt.append("dejan el peaje: " + "; ".join(map(flecha, deja)))
    return " | ".join(txt)


def umbrales_por_tramo(ruta, precio, consumo):
    """Para cada tramo de una ruta fija: Vh a partir del cual conviene la variante con peaje."""
    d = _datos()
    names, c, s = d["names"], d["con_peaje"], d["sin_peaje"]
    k = consumo / 100 * precio
    filas = []
    for idx in range(len(ruta)):
        i, j = ruta[idx - 1], ruta[idx]
        dh = s["horas"][i, j] - c["horas"][i, j]                 # horas que ahorra el peaje
        dcosto = (c["peaje"][i, j] - s["peaje"][i, j]) + k * (c["km"][i, j] - s["km"][i, j])
        if c["peaje"][i, j] < 0.005:
            umbral, nota = 0.0, "la ruta rápida no tiene peaje"
        elif dh <= 1e-6:
            umbral, nota = np.inf, "el peaje no ahorra tiempo: nunca conviene"
        else:
            umbral, nota = max(dcosto / dh, 0.0), ""
        filas.append({
            "desde": names[i], "hasta": names[j], "vias_peaje": c["vias"][i, j],
            "peaje_eur": round(c["peaje"][i, j], 2),
            "ahorro_min": round(60 * dh, 1), "km_extra_sin_peaje": round(s["km"][i, j] - c["km"][i, j], 1),
            "vh_umbral_eur_h": round(umbral, 2), "nota": nota,
        })
    return pd.DataFrame(filas).sort_values("vh_umbral_eur_h")


ESTILO_CAMBIO = {
    "orden": dict(color="#222", ls="-", lw=1.1, label="cambia el orden de las capitales"),
    "sentido": dict(color="#e08a1e", ls="--", lw=1.0, label="cambia el sentido del recorrido"),
    "variante": dict(color="#999", ls=":", lw=1.1, label="un tramo más pasa a pagar peaje"),
}


def _coma(v):
    return f"{v:.2f}".replace(".", ",")


def figura(barrido, cambios, path, vh_ref=None):
    """Arriba: costo de la ruta óptima (tiempo, combustible, peajes). Abajo: tramos con peaje.
    Líneas verticales con su valor en los umbrales; línea discontinua en el valor de referencia."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    x = barrido["vh"]
    ax1.stackplot(x, barrido["tiempo_eur"], barrido["combustible_eur"], barrido["peaje_eur"],
                  labels=["tiempo del vendedor", "combustible", "peajes"],
                  colors=["#2c6fb0", "#8fb8de", "#c0392b"], alpha=0.9)
    ax1.set_ylabel("Costo del recorrido óptimo (€)")
    ax1.grid(alpha=0.25)
    ax1.set_title("Arriba: cuánto cuesta la ruta óptima · Abajo: en cuántos tramos conviene pagar peaje",
                  fontsize=9, color="#444")
    # Escalera con los umbrales exactos (no con la rejilla): cada escalón empieza en su umbral
    cv = cambios[cambios["tipo"] == "variante"].sort_values("vh_umbral")
    xs = [float(x.iloc[0])] + cv["vh_umbral"].tolist() + [float(x.iloc[-1])]
    ys = [int(barrido["tramos_con_peaje"].iloc[0])] + cv["tramos_peaje_despues"].tolist()
    ys = ys + [ys[-1]]
    ax2.step(xs, ys, where="post", color="#c0392b", lw=1.6)
    ax2.set_ylabel("Tramos con peaje")
    ax2.set_xlabel("Valor de la hora del vendedor, Vh (€/h)")
    ax2.set_ylim(-0.5, 11.5)
    ax2.grid(alpha=0.25)

    # Umbrales: una línea por cambio y una etiqueta por valor (escalonadas si están cerca)
    tipos = set(cambios["tipo"])
    for c in cambios.itertuples():
        est = {k: v for k, v in ESTILO_CAMBIO[c.tipo].items() if k != "label"}
        for ax in (ax1, ax2):
            ax.axvline(c.vh_umbral, zorder=1, **est)
    valores = sorted(set(cambios["vh_umbral"].round(2)))
    escalon = lambda v: ys[max(0, np.searchsorted(xs, v, side="right") - 1)]
    nivel, previo = 0, -1e9
    for v in valores:
        nivel = nivel + 1 if v - previo < 4 else 0
        previo = v
        alto = escalon(v) < 5                      # etiqueta lejos de la escalera
        y = (10.9 - 1.15 * nivel) if alto else (0.2 + 1.15 * nivel)
        ax2.text(v + 0.4, y, _coma(v), fontsize=7.5, color="#222", ha="left", va="center",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6))

    if vh_ref is not None:
        c_ref = float(np.interp(vh_ref, x, barrido["costo_eur"]))
        for ax in (ax1, ax2):
            ax.axvline(vh_ref, color="#2e8b57", ls="--", lw=1.4, zorder=3)
        ax1.annotate(f"valor de referencia: {_coma(vh_ref)} €/h\nruta óptima ≈ {f'{c_ref:,.0f}'.replace(',', '.')} €",
                     xy=(vh_ref, c_ref), xytext=(vh_ref + 6, c_ref + 1900), fontsize=8.5, color="#2e8b57",
                     arrowprops=dict(arrowstyle="->", color="#2e8b57"))

    ax1.legend(loc="upper left", fontsize=8.5)
    for t in ("orden", "sentido", "variante"):
        if t in tipos:
            ax2.plot([], [], **ESTILO_CAMBIO[t])
    if vh_ref is not None:
        ax2.plot([], [], color="#2e8b57", ls="--", lw=1.4, label="valor de referencia (INE)")
    ax2.legend(loc="center right", fontsize=8)
    fig.suptitle("Sensibilidad de la ruta óptima al valor de la hora")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def redibujar_figura(carpeta=None, vh_ref=26.18):
    """Vuelve a dibujar la figura a partir de los CSV guardados (sin repetir el barrido)."""
    carpeta = Path(carpeta) if carpeta else SALIDA
    figura(pd.read_csv(carpeta / "barrido_valor_hora.csv"), pd.read_csv(carpeta / "cambios_de_ruta.csv"),
           carpeta / "sensibilidad_valor_hora.png", vh_ref)
    return carpeta / "sensibilidad_valor_hora.png"


def mapa_rutas(rutas, path):
    """Una mini-mapa por cada ruta óptima distinta del barrido (rango de Vh en el título)."""
    d = _datos()
    coords = d["coords"]
    lon, lat = coords["Longitud"].to_numpy(), coords["Latitud"].to_numpy()
    n = len(rutas)
    cols = min(n, 3)
    filas = int(np.ceil(n / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(4.6 * cols, 4.0 * filas), squeeze=False)
    for ax, (rango, r) in zip(axs.flat, rutas):
        te.dibujar_mapa_base(ax, coords)
        ruta = list(r["ruta"])
        for k in range(len(ruta)):
            a, b = ruta[k - 1], ruta[k]
            peaje = (a, b) in r["peaje_arcos"]
            ax.plot([lon[a], lon[b]], [lat[a], lat[b]], color="#c0392b" if peaje else "#2c6fb0",
                    lw=1.8 if peaje else 1.0)
        ax.set_title(f"Vh ∈ {rango}\n{r['tramos_con_peaje']} tramos con peaje · "
                     f"{r['costo_eur']:.0f} € con Vh = {r['vh']:g}", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Rutas óptimas distintas según el valor de la hora (rojo: tramo con peaje)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = te.leer_argumentos(__doc__.splitlines()[0])
    p.add_argument("--vh-min", type=float)
    p.add_argument("--vh-max", type=float)
    p.add_argument("--paso", type=float)
    p.add_argument("--tolerancia", type=float, default=0.01, help="precisión de los umbrales (€/h)")
    p.add_argument("--procesos", type=int, default=None)
    args = p.parse_args()
    cfg = te.config_desde_argumentos(args)
    for v, k in [(args.vh_min, "vh_min"), (args.vh_max, "vh_max"), (args.paso, "vh_paso")]:
        if v is not None:
            cfg["sensibilidad"][k] = v
    ejecutar_barrido(cfg, procesos=args.procesos, tolerancia=args.tolerancia)


def ejecutar_barrido(cfg, procesos=None, tolerancia=0.01, verbose=True, salida=None):
    """Barrido completo del valor de la hora (rango en cfg["sensibilidad"]).

    Guarda tablas y figuras en `salida` (por defecto results/sensibilidad/) y devuelve un dict con:
        barrido (DataFrame), cambios (DataFrame), umbrales_tramo (DataFrame),
        figura_sensibilidad y figura_rutas (rutas de los PNG)."""
    salida = Path(salida) if salida else SALIDA
    procesos = procesos or max(1, min(8, (os.cpu_count() or 2) - 1))
    say = print if verbose else (lambda *a, **k: None)
    s = cfg["sensibilidad"]
    d = _datos()
    te.validar_config(cfg, d["names"])
    names = d["names"]
    precio, cons = cfg["costos"]["precio_combustible"], cfg["costos"]["consumo_l_100km"]
    salida.mkdir(parents=True, exist_ok=True)

    rejilla = np.round(np.arange(s["vh_min"], s["vh_max"] + 1e-9, s["vh_paso"]), 6)
    say(f"Barrido de Vh: {rejilla[0]} → {rejilla[-1]} €/h, paso {s['vh_paso']} "
          f"({len(rejilla)} valores, óptimo exacto en {procesos} procesos)")
    t0 = time.time()
    with ProcessPoolExecutor(procesos, mp_context=_CONTEXTO) as pool:
        res = list(pool.map(resolver_vh, rejilla, [precio] * len(rejilla), [cons] * len(rejilla)))
    say(f"  barrido terminado en {time.time() - t0:.0f} s")

    # Umbrales por bisección entre valores vecinos con soluciones distintas (en paralelo)
    t0 = time.time()
    pendientes = [(a, b, tipo) for a, b in zip(res, res[1:]) for tipo in TIPOS
                  if not misma(a, b, tipo)]
    with ProcessPoolExecutor(procesos, mp_context=_CONTEXTO) as pool:
        futuros = [pool.submit(buscar_cambios, a, b, tipo, precio, cons, tolerancia)
                   for a, b, tipo in pendientes]
        biseccion = [(tipo, lo, hi) for (_, _, tipo), f in zip(pendientes, futuros)
                     for lo, hi in f.result()]
    say(f"  {len(biseccion)} umbrales localizados por bisección en {time.time() - t0:.0f} s")
    cambios = []
    for tipo, lo, hi in sorted(biseccion, key=lambda x: x[1]["vh"]):
        cambios.append({
            "vh_umbral": round((lo["vh"] + hi["vh"]) / 2, 3), "tipo": tipo,
            "costo_antes_eur": round(lo["costo_eur"], 2),
            "tramos_peaje_antes": lo["tramos_con_peaje"],
            "tramos_peaje_despues": hi["tramos_con_peaje"],
            "km_antes": round(lo["km"], 1), "km_despues": round(hi["km"], 1),
            "cambio": describir(lo, hi, tipo, names),
        })
    cambios = pd.DataFrame(cambios, columns=["vh_umbral", "tipo", "costo_antes_eur",
                                             "tramos_peaje_antes", "tramos_peaje_despues",
                                             "km_antes", "km_despues", "cambio"])

    # Tabla del barrido (con un identificador para cada ruta distinta)
    ids, distintas = {}, []
    for r in res:
        clave = (r["arcos"], r["peaje_arcos"])
        if clave not in ids:
            ids[clave] = len(ids) + 1
            distintas.append(r)
        r["id_ruta"] = ids[clave]
    barrido = pd.DataFrame([{k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in r.items()
                             if k not in ("ruta", "arcos", "orden", "peaje_arcos", "peaje_tramos")}
                            for r in res])
    barrido["orden_ciudades"] = [" → ".join(names[i] for i in r["ruta"]) for r in res]
    barrido.to_csv(salida / "barrido_valor_hora.csv", index=False)
    cambios.to_csv(salida / "cambios_de_ruta.csv", index=False)

    # Umbrales por tramo sobre la ruta del valor base
    base = resolver_vh(cfg["costos"]["valor_hora"], precio, cons)
    ut = umbrales_por_tramo(list(base["ruta"]), precio, cons)
    ut.to_csv(salida / "umbrales_por_tramo_ruta_base.csv", index=False)

    figura(barrido, cambios, salida / "sensibilidad_valor_hora.png", cfg["costos"]["valor_hora"])
    rangos = []
    for r in distintas:
        vs = barrido.loc[barrido["id_ruta"] == r["id_ruta"], "vh"]
        rangos.append((f"[{vs.min():g}, {vs.max():g}] €/h", r))
    mapa_rutas(rangos, salida / "rutas_por_valor_hora.png")

    # Resumen en pantalla
    say(f"\nRutas óptimas distintas en el barrido: {len(distintas)}")
    say("Cambios detectados: " + ", ".join(f"{(cambios.tipo == t).sum()} de {t}" for t in TIPOS))
    for c in cambios.itertuples():
        say(f"  Vh ≈ {c.vh_umbral:7.2f} €/h  [{c.tipo:8}]  peajes {c.tramos_peaje_antes}→"
              f"{c.tramos_peaje_despues}, km {c.km_antes}→{c.km_despues}\n      {c.cambio}")
    say(f"\nUmbral por tramo de la ruta óptima con Vh = {cfg['costos']['valor_hora']} €/h "
          "(solo tramos cuya ruta rápida tiene peaje):")
    say(ut[ut["peaje_eur"] > 0].to_string(index=False))
    say(f"\nResultados en {salida}")
    return {"barrido": barrido, "cambios": cambios, "umbrales_tramo": ut,
            "figura_sensibilidad": salida / "sensibilidad_valor_hora.png",
            "figura_rutas": salida / "rutas_por_valor_hora.png"}


if __name__ == "__main__":
    main()

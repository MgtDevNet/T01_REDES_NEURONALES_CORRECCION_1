#!/usr/bin/env python3
"""Rutas con y sin peaje entre las 47 capitales (servidor ORS local).

Para cada par ordenado (i, j), i != j, se piden DOS rutas a OpenRouteService:

  con_peaje : la ruta más rápida, sin restricciones (puede usar autopistas de pago).
  sin_peaje : la ruta más rápida que evita las vías de peaje (avoid_features=tollways).

De cada ruta se guarda: distancia (km), tiempo de viaje (h) y los km recorridos sobre
vías de peaje, desglosados por autopista (p. ej. "AP-6": 41.2), para aplicar después la
tarifa de cada una.

Cómo se detecta el peaje: con el inventario de OpenStreetMap que genera
vias_peaje_osm.py (../data/vias_peaje_osm_nodos.csv: nodos de las vías con toll=yes).
Los puntos de la geometría que devuelve ORS son nodos del mismo mapa, así que un tramo
de la ruta es de peaje si sus dos extremos están a menos de TOLERANCIA_M de nodos de la
misma vía de peaje. NO se usa el indicador de peaje de ORS (extra "waycategory"): marca
como de peaje tramos gratuitos (p. ej. la A-2 entre Lleida y Martorell), y el extra
"tollways" no está disponible para el perfil driving-car.

Requisitos: servidor ORS en localhost:8080 (ver docker-compose.yml) con el mapa
spain-latest.osm.pbf. Ejecución (desde punto_2/parte_2/recoleccion_datos):

    python3 rutas_con_sin_peaje.py

Es reanudable: cada respuesta se añade a un archivo .jsonl y, si se interrumpe,
al volver a ejecutar solo pide los pares que faltan. Al final escribe en ../data/:

    rutas_con_peaje.csv, rutas_sin_peaje.csv   (una fila por par ordenado)
    metadatos_rutas.json                        (fecha del mapa, servidor, etc.)
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

AQUI = Path(__file__).resolve().parent
DATA = AQUI.parent / "data"
BASE = "http://localhost:8080/ors/v2"
VARIANTES = {
    "con_peaje": {},
    "sin_peaje": {"options": {"avoid_features": ["tollways"]}},
}
TOLERANCIA_M = 5                # distancia máx. de un punto de la ruta a un nodo de peaje


def haversine_km(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


class IndicePeajes:
    """Búsqueda rápida del nodo de peaje más cercano (árbol k-d en coordenadas planas)."""

    def __init__(self, ruta_csv):
        nodos = pd.read_csv(ruta_csv)
        self.refs = nodos["ref"].to_numpy()
        self.arbol = cKDTree(self._plano(nodos[["lon", "lat"]].to_numpy()))

    @staticmethod
    def _plano(lonlat):
        lonlat = np.asarray(lonlat, float)[:, :2]
        return np.c_[lonlat[:, 0] * 111_320 * np.cos(np.radians(40.0)), lonlat[:, 1] * 110_574]

    def ref_de_cada_punto(self, coords):
        """Para cada punto, la vía de peaje a la que pertenece (o None)."""
        dist, idx = self.arbol.query(self._plano(coords), distance_upper_bound=TOLERANCIA_M)
        return [self.refs[i] if d <= TOLERANCIA_M else None for d, i in zip(dist, idx)]


PEAJES = None                    # se carga en main()


def km_peaje_por_via(feature):
    """Suma los km de la ruta que van por vías de peaje, agrupados por vía."""
    coords = feature["geometry"]["coordinates"]
    refs = PEAJES.ref_de_cada_punto(coords)
    por_via, km_geo = {}, 0.0
    for p, q, rp, rq in zip(coords, coords[1:], refs, refs[1:]):
        d = haversine_km(p, q)
        km_geo += d
        if rp is not None and rp == rq:
            por_via[rp] = por_via.get(rp, 0.0) + d
    # La geometría (haversine) y la distancia de ORS difieren un poco: se reescala.
    escala = feature["properties"]["summary"]["distance"] / 1000 / (km_geo or 1.0)
    return {k: round(v * escala, 3) for k, v in sorted(por_via.items())}


def pedir_ruta(sesion, origen, destino, variante):
    cuerpo = {
        "coordinates": [origen, destino],
        "preference": "fastest",
        "instructions": False,
        **VARIANTES[variante],
    }
    for intento in range(3):
        try:
            r = sesion.post(f"{BASE}/directions/driving-car/geojson", json=cuerpo, timeout=120)
            if r.status_code == 200:
                f = r.json()["features"][0]
                via = km_peaje_por_via(f)
                return {
                    "km": f["properties"]["summary"]["distance"] / 1000,
                    "horas": f["properties"]["summary"]["duration"] / 3600,
                    "km_peaje": round(sum(via.values()), 3),
                    "peaje_por_via": via,
                    "error": "",
                }
            error = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code < 500:
                break                                   # error del pedido: no reintentar
        except (requests.RequestException, ValueError, KeyError) as e:
            error = str(e)
        time.sleep(2 * (intento + 1))
    return {"km": None, "horas": None, "km_peaje": None, "peaje_por_via": {}, "error": error}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hilos", type=int, default=6, help="peticiones simultáneas al servidor")
    ap.add_argument("--cache", default=str(AQUI / "rutas_cache.jsonl"))
    ap.add_argument("--prueba", action="store_true", help="solo 3 pares, para probar")
    args = ap.parse_args()

    try:
        estado = requests.get(f"{BASE}/health", timeout=10).json().get("status")
    except requests.RequestException as e:
        sys.exit(f"No se pudo conectar con ORS ({e}). ¿Está corriendo? -> docker compose up -d")
    if estado != "ready":
        sys.exit(f"ORS aún no está listo (status={estado}). Espera a que termine el grafo.")

    global PEAJES
    nodos_peaje = DATA / "vias_peaje_osm_nodos.csv"
    if not nodos_peaje.exists():
        sys.exit(f"Falta {nodos_peaje}. Ejecuta antes: python3 vias_peaje_osm.py")
    PEAJES = IndicePeajes(nodos_peaje)

    coords = pd.read_csv(DATA / "coordenadas.csv", encoding="utf-8-sig")
    nombres = coords["Capital"].tolist()
    lonlat = coords[["Longitud", "Latitud"]].values.tolist()
    n = len(nombres)

    pendientes = [(i, j, v) for i in range(n) for j in range(n) if i != j for v in VARIANTES]
    if args.prueba:
        pendientes = [(0, 1, v) for v in VARIANTES] + [(29, 7, v) for v in VARIANTES]

    cache = Path(args.cache)
    hechos = {}
    if cache.exists() and not args.prueba:
        for linea in cache.read_text(encoding="utf-8").splitlines():
            d = json.loads(linea)
            if not d["error"]:
                hechos[(d["i"], d["j"], d["variante"])] = d
    pendientes = [p for p in pendientes if p not in hechos]
    print(f"Pares ya calculados: {len(hechos)} · pendientes: {len(pendientes)}")

    t0 = time.time()
    with requests.Session() as sesion, ThreadPoolExecutor(args.hilos) as pool, \
            open(cache, "a", encoding="utf-8") as out:
        futuros = {pool.submit(pedir_ruta, sesion, lonlat[i], lonlat[j], v): (i, j, v)
                   for i, j, v in pendientes}
        for k, fut in enumerate(as_completed(futuros), 1):
            i, j, v = futuros[fut]
            d = {"i": i, "j": j, "origen": nombres[i], "destino": nombres[j], "variante": v,
                 **fut.result()}
            if args.prueba:
                print(json.dumps(d, ensure_ascii=False, indent=1))
                continue
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
            out.flush()
            hechos[(i, j, v)] = d
            if k % 200 == 0 or k == len(pendientes):
                print(f"  {k}/{len(pendientes)}  ({time.time() - t0:.0f} s)")
    if args.prueba:
        return

    errores = [d for d in hechos.values() if d["error"]]
    faltan = n * (n - 1) * len(VARIANTES) - len(hechos)
    if errores or faltan:
        sys.exit(f"Quedan {faltan} rutas sin calcular y {len(errores)} con error. "
                 "Vuelve a ejecutar el script para reintentarlas.")

    for v in VARIANTES:
        filas = sorted((d for d in hechos.values() if d["variante"] == v),
                       key=lambda d: (d["i"], d["j"]))
        df = pd.DataFrame([{
            "origen": d["origen"], "destino": d["destino"],
            "km": round(d["km"], 3), "horas": round(d["horas"], 5),
            "km_peaje": d["km_peaje"],
            "peaje_por_via": json.dumps(d["peaje_por_via"], ensure_ascii=False),
        } for d in filas])
        df.to_csv(DATA / f"rutas_{v}.csv", index=False, encoding="utf-8")
        print(f"Guardado {DATA / f'rutas_{v}.csv'} ({len(df)} filas)")

    pbf = AQUI / "ors-files" / "spain-latest.osm.pbf"
    fecha_desc = AQUI / "ors-files" / "FECHA_DESCARGA.txt"
    meta = {
        "fuente": "OpenRouteService v8.1.1 (servidor local, docker-compose.yml), perfil driving-car",
        "endpoint": "/ors/v2/directions/driving-car/geojson",
        "preferencia": "fastest",
        "deteccion_peaje": f"nodos OSM toll=yes (vias_peaje_osm_nodos.csv), tolerancia {TOLERANCIA_M} m",
        "variantes": {"con_peaje": "sin restricciones",
                      "sin_peaje": "options.avoid_features = ['tollways']"},
        "mapa": "Geofabrik spain-latest.osm.pbf (OpenStreetMap)",
        "fecha_mapa": (date.fromtimestamp(pbf.stat().st_mtime).isoformat()
                       if pbf.exists() else None),
        "fecha_descarga_mapa": (fecha_desc.read_text().strip() if fecha_desc.exists() else None),
        "fecha_consulta": date.today().isoformat(),
        "n_capitales": n,
        "pares_ordenados": n * (n - 1),
    }
    (DATA / "metadatos_rutas.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print("Guardado metadatos_rutas.json")


if __name__ == "__main__":
    main()

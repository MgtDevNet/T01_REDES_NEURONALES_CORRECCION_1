#!/usr/bin/env python3
"""Inventario de las vías de peaje según OpenStreetMap (mapa spain-latest.osm.pbf).

Recorre el mapa y extrae todas las vías para coches (highway=motorway, trunk, primary...)
con la etiqueta toll=yes. Las agrupa por referencia (AP-6, AP-9, C-33...) y suma sus km.
Así se puede comparar lo que OSM considera "de peaje" con la lista oficial de autopistas
de peaje vigentes en 2026 y detectar etiquetas desactualizadas.

Ejecución (desde punto_2/parte_2/recoleccion_datos; requiere  pip install osmium):

    python3 vias_peaje_osm.py

Salidas en ../data/:
    vias_peaje_osm.csv                una fila por referencia: km de calzada con toll=yes
    vias_peaje_osm_nodos_crudo.csv    puntos (lon, lat) de cada vía, con la ref de OSM
    vias_peaje_osm_nodos.csv          los mismos puntos con la vía NORMALIZADA (ver
                                      normalizar()): es el que usa rutas_con_sin_peaje.py

Para repetir solo la normalización (sin volver a leer el mapa, ~13 min):
    python3 vias_peaje_osm.py --solo-normalizar
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

AQUI = Path(__file__).resolve().parent
PBF = AQUI / "ors-files" / "spain-latest.osm.pbf"
DATA = AQUI.parent / "data"
VIAS_COCHE = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
              "secondary", "secondary_link", "tertiary", "unclassified", "residential"}


def haversine_km(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


# Variantes de nombre en OSM que pertenecen a la misma autopista de peaje.
ALIAS = {
    "AP-9M": "AP-9", "AP-9 M": "AP-9", "AP-9F": "AP-9", "AP-9SF": "AP-9", "AP-9FS": "AP-9",
    "A 9": "AP-9", "Autoestrada Central Galega": "AP-53",
    "A-66": "AP-66", "Enlace de La Magdalena": "AP-66", "Enlace Barrios de Luna": "AP-66",
    "N-622": "AP-68",                                   # enlace Vitoria–AP-68 (Álava)
    "Bizkaiko Golkoko Autobidea": "AP-8", "Kantauriko Autobidea": "AP-8",
    "BI-626": "AP-8", "BI-627": "AP-8", "AP-8 Supersur": "AP-8",
    "Variante Sur Metropolitana": "AP-8", "Gerediaga - Elorrio Autobidea": "AP-8",
    "GI-20": "AP-8", "DO": "AP-8",                      # accesos de Donostia (Bidegi)
    "PA-34": "AP-15",
}
# AP-68 tras el fin de la concesión estatal (10-nov-2026): solo cobran Bizkaia y Álava.
# Límites aproximados por latitud sobre el trazado (la AP-68 va casi norte-sur):
#   Bizkaia: Bilbao -> Ziorraga (~22,3 km)  |  Álava: Ziorraga -> límite con Burgos.
LAT_BIZKAIA_ALAVA = 43.07
LAT_ALAVA_BURGOS = 42.70
DIST_SIN_REF_M = 3000          # un nodo sin ref se asigna a la vía con nombre más cercana


def normalizar(nodos):
    """Unifica alias, asigna los nodos sin ref y divide la AP-68 por territorio."""
    n = nodos.copy()
    n["ref"] = n["ref"].replace(ALIAS)
    xy = np.c_[n["lon"] * 111_320 * np.cos(np.radians(40.0)), n["lat"] * 110_574]
    con_ref = ~n["ref"].isin(["(sin ref)"]) & n["ref"].str.match(r"^(AP|R|C|AG|M)-\d")
    arbol = cKDTree(xy[con_ref.to_numpy()])
    refs_validas = n.loc[con_ref, "ref"].to_numpy()
    d, i = arbol.query(xy[~con_ref.to_numpy()])
    n.loc[~con_ref, "ref"] = np.where(d <= DIST_SIN_REF_M, refs_validas[i], "(descartado)")
    ap68 = n["ref"] == "AP-68"
    n.loc[ap68 & (n["lat"] >= LAT_BIZKAIA_ALAVA), "ref"] = "AP-68 Bizkaia"
    n.loc[ap68 & (n["lat"] < LAT_BIZKAIA_ALAVA) & (n["lat"] >= LAT_ALAVA_BURGOS), "ref"] = "AP-68 Álava"
    n.loc[ap68 & (n["lat"] < LAT_ALAVA_BURGOS), "ref"] = "(descartado)"   # tramo estatal, libre
    return n[n["ref"] != "(descartado)"].drop_duplicates()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solo-normalizar", action="store_true",
                    help="reutiliza vias_peaje_osm_nodos_crudo.csv sin leer el mapa")
    args = ap.parse_args()
    crudo = DATA / "vias_peaje_osm_nodos_crudo.csv"
    if args.solo_normalizar:
        n = normalizar(pd.read_csv(crudo))
        n.to_csv(DATA / "vias_peaje_osm_nodos.csv", index=False, encoding="utf-8")
        print(n["ref"].value_counts().to_string())
        return

    import osmium                                    # solo hace falta para leer el mapa
    # Paso 1: vías con toll=yes (sin coordenadas todavía)
    vias = {}
    for w in osmium.FileProcessor(str(PBF), osmium.osm.WAY):
        if w.tags.get("toll") == "yes" and w.tags.get("highway") in VIAS_COCHE:
            ref = w.tags.get("ref") or w.tags.get("name") or "(sin ref)"
            vias[w.id] = (ref.split(";")[0].strip(), w.tags.get("highway"),
                          [n.ref for n in w.nodes])
    print(f"Vías con toll=yes: {len(vias)}")

    # Paso 2: coordenadas solo de los nodos de esas vías
    nodos = {n for _, _, ns in vias.values() for n in ns}
    pos = {}
    for n in osmium.FileProcessor(str(PBF), osmium.osm.NODE):
        if n.id in nodos:
            pos[n.id] = (n.location.lon, n.location.lat)

    km = defaultdict(float)
    tipo = defaultdict(set)
    puntos = []
    for wid, (ref, hw, ns) in vias.items():
        pts = [pos[n] for n in ns if n in pos]
        km[ref] += sum(haversine_km(p, q) for p, q in zip(pts, pts[1:]))
        tipo[ref].add(hw)
        puntos += [(ref, round(p[0], 6), round(p[1], 6)) for p in pts]

    df = (pd.DataFrame({"ref": list(km), "km_calzada": [round(km[r], 2) for r in km],
                        "tipos_highway": [",".join(sorted(tipo[r])) for r in km]})
          .sort_values("km_calzada", ascending=False))
    df.to_csv(DATA / "vias_peaje_osm.csv", index=False, encoding="utf-8")
    nodos = pd.DataFrame(puntos, columns=["ref", "lon", "lat"]).drop_duplicates()
    nodos.to_csv(crudo, index=False, encoding="utf-8")
    normalizar(nodos).to_csv(DATA / "vias_peaje_osm_nodos.csv", index=False, encoding="utf-8")
    print(df.head(60).to_string(index=False))


if __name__ == "__main__":
    main()

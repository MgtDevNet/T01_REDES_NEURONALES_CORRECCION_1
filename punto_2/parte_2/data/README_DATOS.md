# Datos del punto 2.2: fuentes y fechas de consulta

Todos los datos que necesita `src/tsp_espana.py` están en esta carpeta. **La ejecución no
necesita internet ni ninguna API.** Los scripts de `../recoleccion_datos/` documentan cómo se
obtuvieron y permiten regenerarlos.

## Modelo de costo

Para cada par ordenado de capitales (i → j) hay dos rutas por carretera:

| Variante | Qué es |
|---|---|
| `con_peaje` | la ruta más rápida, sin restricciones (puede usar autopistas de pago) |
| `sin_peaje` | la ruta más rápida que **evita** las vías de peaje |

    C_ij(Vh) = min sobre las dos variantes de   Vh · horas  +  k · km  +  peaje
    k = consumo/100 · precio del combustible   (€/km)

El vendedor elige en cada tramo pagar el peaje o ir por la red libre, así que la ruta óptima
depende del valor de la hora `Vh`.

## Archivos

| Archivo | Contenido | Origen |
|---|---|---|
| `coordenadas.csv` | Capital, Longitud, Latitud de las 47 capitales peninsulares | `../recoleccion_datos/datos_capitales.xlsx` |
| `rutas_con_peaje.csv` | 2162 pares: km, horas, km de peaje y desglose por autopista (`peaje_por_via`, JSON) | `rutas_con_sin_peaje.py` |
| `rutas_sin_peaje.csv` | igual, para la ruta que evita los peajes | `rutas_con_sin_peaje.py` |
| `metadatos_rutas.json` | servidor, mapa, fechas y método de detección del peaje | `rutas_con_sin_peaje.py` |
| `tarifas_peaje_2026.csv` | €/km de cada vía de peaje, con recorrido de referencia, criterio, fuente, URL y fecha | elaboración propia (ver abajo) |
| `vias_peaje_osm.csv` | inventario de vías con `toll=yes` en OSM y sus km de calzada | `vias_peaje_osm.py` |
| `vias_peaje_osm_nodos_crudo.csv` | nodos (lon, lat) de esas vías con la referencia original de OSM | `vias_peaje_osm.py` |
| `vias_peaje_osm_nodos.csv` | los mismos nodos con la vía normalizada (alias unificados, AP-68 dividida por territorio) | `vias_peaje_osm.py` |
| `contorno_espana.csv` | contorno de España peninsular, Portugal, Francia y Andorra (solo para los mapas) | Natural Earth |

## Fuentes y fechas de consulta

| Dato | Valor | Fuente | Fecha |
|---|---|---|---|
| Red de carreteras | OpenStreetMap, extracto de España | Geofabrik, `spain-latest.osm.pbf` (https://download.geofabrik.de/europe/spain.html) | mapa del 17-11-2026, descargado el 18-11-2026 |
| Distancias y tiempos | rutas más rápidas, perfil `driving-car` | OpenRouteService v8.1.1 en servidor local (`docker-compose.yml`, `ors-config.yml`) | 19-11-2026 |
| Tramos de peaje | nodos de vías con `toll=yes` | OpenStreetMap (mismo mapa) | 19-11-2026 |
| Tarifas de peaje | €/km por autopista, vehículo ligero | SEITT, Bidegi, AEA y prensa; ver `tarifas_peaje_2026.csv` | 19-11-2026 |
| AP-68 | gratuita en La Rioja, Navarra y Aragón desde el 11-11-2026; Bizkaia 3,35 €; Álava 0,90 € | GasteizBerri, Radio Llodio, Moncloa.com (2026) | 19-11-2026 |
| Vehículo | Toyota Corolla Hybrid 140H, gasolina, 4,7 L/100 km (WLTP, extremo alto) | Toyota España (https://www.toyota.es/coches/corolla) | 05-10-2026 |
| Combustible | gasolina 95: 1,799 €/L | Gasolinera Hoy, datos de MITECO (https://gasolinerahoy.es/) | 05-10-2026 |
| Valor de la hora | 26,18 €/h: coste laboral por hora efectiva, 2.º trimestre de 2026 | INE, Encuesta Trimestral de Coste Laboral (https://www.ine.es/dyngs/Prensa/ETCL2T26.htm) | publicado el 17-09-2026 |
| Contorno del mapa | países, escala 1:50m, v5.1.2 | Natural Earth (dominio público) | 19-11-2026 |

## Cómo se detecta el peaje de una ruta

1. `vias_peaje_osm.py` lee el mapa y guarda los nodos de todas las vías para coches con `toll=yes`.
2. `rutas_con_sin_peaje.py` pide a ORS cada ruta con su geometría. Un segmento de la ruta es de
   peaje si sus dos extremos están a menos de 5 m de nodos de la **misma** vía de peaje.
3. El peaje del tramo es `Σ km en cada vía × tarifa €/km de esa vía`.

No se usa el indicador de peaje propio de ORS (extra `waycategory`) porque marca como de peaje
tramos gratuitos: por ejemplo, 161,6 km de "peaje" en Madrid→Barcelona, sobre la A-2 entre Lleida y Martorell, en Madrid→Barcelona.

## Supuestos y limitaciones (deben aparecer en el informe)

- **Tarifa media por km.** El precio de cada autopista se aproxima como precio del recorrido
  de referencia / su longitud. El peaje real depende de la entrada y la salida concretas; para
  los tramos entre capitales la aproximación es razonable, pero no es la tarifa exacta de cada
  trayecto.
- **Vehículo ligero, pago en efectivo, de día, sin bonificaciones.** El vendedor viaja de día y
  no es usuario habitual, así que no se aplican la gratuidad nocturna de SEITT (0:00–6:00) ni
  los descuentos por recurrencia (Bidegi, Audenasa, AP-9). En la AP-51 y la AP-61 se usa la tarifa
  "normal", y en la AP-7 la de temporada baja (noviembre).
- **AP-68 vasca.** Las tarifas forales posteriores al 11-11-2026 vienen de prensa local: aún no
  estaban publicadas en el BOTHA ni en el BOB en la fecha de consulta. El límite Bizkaia/Álava
  (lat. 43,07, Ziorraga) y Álava/Burgos (lat. 42,70) es aproximado.
- **Tiempos de viaje.** Son los de ORS: velocidad por tipo de vía y sin tráfico real. Son
  iguales a cualquier hora del día.
- **Combustible.** Precio único nacional de un día concreto; consumo de homologación, no medido.
- **Asimetría.** Ida y vuelta pueden medir y costar distinto. El modelo lo respeta (TSP asimétrico).

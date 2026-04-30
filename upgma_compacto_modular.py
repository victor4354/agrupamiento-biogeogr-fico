#!/usr/bin/env python3
"""
upgma_compacto_modular.py

Pipeline de clustering jerárquico con múltiples índices de similitud.
Soporta tres métodos de linkage: single, complete y average (UPGMA).

Uso:
  python upgma_compacto_modular.py -i datos.xlsx -o resultados

Opciones principales:
  --sim-index       índice de similitud (jaccard, simpson, sorensen, ochiai,
                    braun-blanquet, fager, kulezynski, correlation, baroni)
  --linkage-method  método de linkage (single, complete, average)
  --umbral-corte N  corta el dendrograma al N% de similitud y asigna grupos
  --export-distance escribe la matriz de distancia (1 - similitud)
  --percent         escala similitud a 0..100 en el CSV
  --min-species N   filtra estados con menos de N especies presentes
  --list-sheets     lista las hojas del Excel y sale

Índices de similitud disponibles:
  jaccard        |A ∩ B| / |A ∪ B|
  simpson        |A ∩ B| / min(|A|, |B|)
  sorensen       2 × |A ∩ B| / (|A| + |B|)
  ochiai         |A ∩ B| / sqrt(|A| × |B|)
  braun-blanquet |A ∩ B| / max(|A|, |B|)
  fager          |A ∩ B| / (sqrt(|A|x|B|) - 0.5 x max(|A|,|B|))  * no acotado
  kulezynski     (|A∩B| x |A∪B|) / (2 x |A| x |B|)               * no acotado
  correlation    |A ∩ B| / (|A| × |B|)                            * no acotado
  baroni         (sqrt(|A∩B| x |(A∪B)c|)+|A∩B|)/(sqrt(...)+|A∪B|)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ===========================================================================
#  I/O y normalizacion
# ===========================================================================

def read_input_excel(path: Path, sheet: str | None = None) -> pd.DataFrame:
    """Lee el archivo Excel y devuelve un DataFrame."""
    logger.info(f"Leyendo datos desde: {path}")

    sheet_name = 0 if sheet is None else sheet
    try:
        if sheet is not None:
            s = str(sheet).strip()
            sheet_name = int(s) if s.isdigit() else sheet
    except Exception:
        sheet_name = sheet

    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except ValueError as e:
        raise ValueError(
            f"Error leyendo Excel: {e}\n"
            "Asegurate de que --sheet sea un nombre de hoja valido o un indice (p. ej. 0)."
        ) from e

    if isinstance(df, dict):
        first_key = next(iter(df.keys()))
        logger.warning(f"pd.read_excel devolvio varias hojas; usando la primera: '{first_key}'")
        df = df[first_key]

    if not hasattr(df, 'shape'):
        raise ValueError(f"El contenido leido de {path} no parece un DataFrame. Tipo: {type(df)}")

    logger.info(f"Datos leidos: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def detect_columns(
    df: pd.DataFrame,
    species_col: str | None,
    state_col: str | None,
) -> Tuple[str, str]:
    """Detecta las columnas de especie y estado si no se pasan exactas."""
    cols_originales = list(df.columns)
    lower_map = {c.lower(): c for c in cols_originales}
    cols_lower = list(lower_map.keys())

    def buscar(candidatos: List[str]) -> str | None:
        for pat in candidatos:
            for low in cols_lower:
                if pat in low:
                    return lower_map[low]
        return None

    if species_col is None or species_col not in cols_originales:
        candidate = buscar(['espec', 'species', 'taxon', 'sp'])
        if candidate:
            species_col = candidate
            logger.info(f"Columna 'species' detectada como: {species_col}")
        else:
            raise ValueError(
                "No pude detectar la columna de 'especie'. "
                "Pasa --species con el nombre exacto."
            )

    if state_col is None or state_col not in cols_originales:
        candidate = buscar(['estado', 'state', 'region', 'site', 'loc'])
        if candidate:
            state_col = candidate
            logger.info(f"Columna 'state' detectada como: {state_col}")
        else:
            raise ValueError(
                "No pude detectar la columna de 'estado'. "
                "Pasa --state con el nombre exacto."
            )

    return species_col, state_col


def clean_state_values(series: pd.Series) -> pd.Series:
    """Limpia y normaliza los valores de la columna 'estado'."""
    s = series.astype(str).str.strip()
    vacios = {'nd', 'na', 'n/a', 'nan', '', 'none', 'nan.'}
    s_lower = s.str.lower()
    return s.where(~s_lower.isin(vacios), other=pd.NA)


# ===========================================================================
#  Matriz presencia / ausencia
# ===========================================================================

def build_presence_absence(
    df: pd.DataFrame,
    species_col: str,
    state_col: str,
) -> pd.DataFrame:
    """Construye la matriz presencia/ausencia (especies x estados) con 0/1."""
    logger.info("Construyendo matriz presencia/ausencia...")
    df = df.copy()
    df[state_col] = clean_state_values(df[state_col])
    df = df.dropna(subset=[state_col, species_col])
    pa = pd.crosstab(df[species_col], df[state_col])
    pa = (pa > 0).astype(int)
    logger.info(f"Matriz PA: {pa.shape[0]} especies x {pa.shape[1]} estados")
    return pa


# ===========================================================================
#  Indices de similitud vectorizados (NumPy)
# ===========================================================================

def compute_jaccard(pa: pd.DataFrame, percent: bool = False) -> pd.DataFrame:
    """Jaccard = |A ∩ B| / |A ∪ B|   rango [0, 1]"""
    logger.info("Calculando similitud Jaccard...")
    X = pa.T.values.astype(int)
    inter = X.dot(X.T).astype(float)
    counts = X.sum(axis=1).astype(float)
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(union > 0, inter / union, np.nan)
    df = pd.DataFrame(mat, index=pa.columns, columns=pa.columns)
    return df * 100.0 if percent else df


def compute_simpson(pa: pd.DataFrame, percent: bool = False) -> pd.DataFrame:
    """Simpson = |A ∩ B| / min(|A|, |B|)   rango [0, 1]"""
    logger.info("Calculando similitud Simpson...")
    X = pa.T.values.astype(int)
    inter = X.dot(X.T).astype(float)
    counts = X.sum(axis=1).astype(float)
    min_c = np.minimum(counts[:, None], counts[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(min_c > 0, inter / min_c, np.nan)
    df = pd.DataFrame(mat, index=pa.columns, columns=pa.columns)
    return df * 100.0 if percent else df


def compute_sorensen(pa: pd.DataFrame, percent: bool = False) -> pd.DataFrame:
    """Sorensen-Dice = 2|A ∩ B| / (|A| + |B|)   rango [0, 1]"""
    logger.info("Calculando similitud Sorensen-Dice...")
    X = pa.T.values.astype(int)
    inter = X.dot(X.T).astype(float)
    counts = X.sum(axis=1).astype(float)
    sum_c = counts[:, None] + counts[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(sum_c > 0, 2.0 * inter / sum_c, np.nan)
    df = pd.DataFrame(mat, index=pa.columns, columns=pa.columns)
    return df * 100.0 if percent else df


def compute_ochiai(pa: pd.DataFrame, percent: bool = False) -> pd.DataFrame:
    """Ochiai = |A ∩ B| / sqrt(|A| x |B|)   rango [0, 1]"""
    logger.info("Calculando similitud Ochiai...")
    X = pa.T.values.astype(int)
    inter = X.dot(X.T).astype(float)
    counts = X.sum(axis=1).astype(float)
    geo = np.sqrt(counts[:, None] * counts[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(geo > 0, inter / geo, np.nan)
    df = pd.DataFrame(mat, index=pa.columns, columns=pa.columns)
    return df * 100.0 if percent else df


def compute_braun_blanquet(pa: pd.DataFrame, percent: bool = False) -> pd.DataFrame:
    """Braun-Blanquet = |A ∩ B| / max(|A|, |B|)   rango [0, 1]"""
    logger.info("Calculando similitud Braun-Blanquet...")
    X = pa.T.values.astype(int)
    inter = X.dot(X.T).astype(float)
    counts = X.sum(axis=1).astype(float)
    max_c = np.maximum(counts[:, None], counts[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(max_c > 0, inter / max_c, np.nan)
    df = pd.DataFrame(mat, index=pa.columns, columns=pa.columns)
    return df * 100.0 if percent else df


def compute_fager(pa: pd.DataFrame, percent: bool = False) -> pd.DataFrame:
    """Fager = |A∩B| / (sqrt(|A|x|B|) - 0.5 x max(|A|,|B|))   * NO acotado a [0,1]"""
    logger.info("Calculando similitud Fager...")
    X = pa.T.values.astype(int)
    inter = X.dot(X.T).astype(float)
    counts = X.sum(axis=1).astype(float)
    geo = np.sqrt(counts[:, None] * counts[None, :])
    max_c = np.maximum(counts[:, None], counts[None, :])
    denom = geo - 0.5 * max_c
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(denom > 0, inter / denom, np.nan)
    df = pd.DataFrame(mat, index=pa.columns, columns=pa.columns)
    return df * 100.0 if percent else df


def compute_kulezynski(pa: pd.DataFrame, percent: bool = False) -> pd.DataFrame:
    """Kulezynski = (|A∩B| x |A∪B|) / (2 x |A| x |B|)   * NO acotado a [0,1]"""
    logger.info("Calculando similitud Kulezynski...")
    X = pa.T.values.astype(int)
    inter = X.dot(X.T).astype(float)
    counts = X.sum(axis=1).astype(float)
    union = counts[:, None] + counts[None, :] - inter
    denom = 2.0 * counts[:, None] * counts[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(denom > 0, (inter * union) / denom, np.nan)
    df = pd.DataFrame(mat, index=pa.columns, columns=pa.columns)
    return df * 100.0 if percent else df


def compute_correlation_ratio(pa: pd.DataFrame, percent: bool = False) -> pd.DataFrame:
    """Radio de Correlacion = |A∩B| / (|A| x |B|)   * NO acotado a [0,1]"""
    logger.info("Calculando Radio de Correlacion...")
    X = pa.T.values.astype(int)
    inter = X.dot(X.T).astype(float)
    counts = X.sum(axis=1).astype(float)
    prod = counts[:, None] * counts[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(prod > 0, inter / prod, np.nan)
    df = pd.DataFrame(mat, index=pa.columns, columns=pa.columns)
    return df * 100.0 if percent else df


def compute_baroni_urbani_buser(pa: pd.DataFrame, percent: bool = False) -> pd.DataFrame:
    """Baroni-Urbani-Buser = (sqrt(|A∩B| x |(A∪B)c|)+|A∩B|) / (sqrt(...)+|A∪B|)   rango [0, 1]"""
    logger.info("Calculando similitud Baroni-Urbani-Buser...")
    X = pa.T.values.astype(int)
    n_species = pa.shape[0]
    inter = X.dot(X.T).astype(float)
    counts = X.sum(axis=1).astype(float)
    union = counts[:, None] + counts[None, :] - inter
    complement = n_species - union
    sqrt_term = np.sqrt(np.maximum(inter * complement, 0))
    numer = sqrt_term + inter
    denom = sqrt_term + union
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(denom > 0, numer / denom, np.nan)
    df = pd.DataFrame(mat, index=pa.columns, columns=pa.columns)
    return df * 100.0 if percent else df


def compute_similarity(
    pa: pd.DataFrame,
    index_name: str = 'jaccard',
    percent: bool = False,
) -> pd.DataFrame:
    """Despachador central: calcula el indice de similitud especificado."""
    dispatch = {
        'jaccard':             compute_jaccard,
        'simpson':             compute_simpson,
        'sorensen':            compute_sorensen,
        'dice':                compute_sorensen,
        'sorensen-dice':       compute_sorensen,
        'ochiai':              compute_ochiai,
        'braun-blanquet':      compute_braun_blanquet,
        'braun_blanquet':      compute_braun_blanquet,
        'bb':                  compute_braun_blanquet,
        'fager':               compute_fager,
        'kulezynski':          compute_kulezynski,
        'correlation':         compute_correlation_ratio,
        'correlation-ratio':   compute_correlation_ratio,
        'radio':               compute_correlation_ratio,
        'baroni':              compute_baroni_urbani_buser,
        'baroni-urbani-buser': compute_baroni_urbani_buser,
        'bub':                 compute_baroni_urbani_buser,
    }
    key = index_name.lower()
    if key not in dispatch:
        raise ValueError(
            f"Indice '{index_name}' no reconocido. "
            "Opciones: jaccard, simpson, sorensen, ochiai, braun-blanquet, "
            "fager, kulezynski, correlation, baroni"
        )
    return dispatch[key](pa, percent)


# ===========================================================================
#  Clustering jerarquico (UPGMA / single / complete)
# ===========================================================================

def hierarchical_linkage_from_similarity(
    sim_df: pd.DataFrame,
    method: str = 'average',
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]], List[str]]:
    """
    Clustering jerarquico a partir de una MATRIZ DE SIMILITUD.

    Retorna matriz de linkage compatible con scipy:
        [cluster_a, cluster_b, distancia, tamano_nuevo_cluster]

    La distancia se define como: distancia = 1 - similitud

    method: 'single' | 'complete' | 'average' (UPGMA)
    """
    method = method.lower()
    if method == 'upgma':
        method = 'average'
    if method not in ('single', 'complete', 'average'):
        raise ValueError(f"Metodo '{method}' no reconocido. Opciones: single, complete, average")

    labels = list(sim_df.columns)
    n = len(labels)

    if n == 0:
        return np.array([]), {}, []
    if n == 1:
        return np.array([]), {0: {'members': [labels[0]], 'height': 0.0, 'size': 1}}, labels

    S = sim_df.values.astype(float).copy()
    np.fill_diagonal(S, np.nan)

    clusters: Dict[int, Dict[str, Any]] = {
        i: {'members': [labels[i]], 'size': 1, 'height': 0.0}
        for i in range(n)
    }
    active = list(range(n))
    next_id = n
    linkage_rows: List[List[float]] = []

    for _ in range(n - 1):
        best_sim = -np.inf
        best_pair = (None, None)
        for a_idx in range(len(active)):
            for b_idx in range(a_idx + 1, len(active)):
                ia, ib = active[a_idx], active[b_idx]
                val = S[ia, ib]
                if not np.isnan(val) and val > best_sim:
                    best_sim = val
                    best_pair = (ia, ib)

        if best_pair[0] is None:
            best_pair = (active[0], active[1])
            best_sim = np.nan

        i_idx, j_idx = best_pair
        size_i = clusters[i_idx]['size']
        size_j = clusters[j_idx]['size']
        new_size = size_i + size_j
        distance = 1.0 - best_sim if not np.isnan(best_sim) else np.nan

        linkage_rows.append([float(i_idx), float(j_idx), float(distance), float(new_size)])

        clusters[next_id] = {
            'members': clusters[i_idx]['members'] + clusters[j_idx]['members'],
            'size': new_size,
            'height': distance,
        }

        if next_id >= S.shape[0]:
            new_dim = next_id + 1
            S_new = np.full((new_dim, new_dim), np.nan, dtype=float)
            S_new[:S.shape[0], :S.shape[1]] = S
            S = S_new

        for k in active:
            if k in (i_idx, j_idx):
                continue
            s_ik = S[i_idx, k]
            s_jk = S[j_idx, k]

            if method == 'single':
                new_s = np.nanmax([s_ik, s_jk])
            elif method == 'complete':
                new_s = np.nanmin([s_ik, s_jk])
            else:  # average / UPGMA
                if np.isnan(s_ik) and np.isnan(s_jk):
                    new_s = np.nan
                elif np.isnan(s_ik):
                    new_s = s_jk
                elif np.isnan(s_jk):
                    new_s = s_ik
                else:
                    new_s = (s_ik * size_i + s_jk * size_j) / new_size

            S[next_id, k] = new_s
            S[k, next_id] = new_s

        S[next_id, next_id] = np.nan
        active.remove(i_idx)
        active.remove(j_idx)
        active.append(next_id)
        next_id += 1

    return np.array(linkage_rows), clusters, labels


# ===========================================================================
#  Corte del dendrograma → grupos biogeograficos
# ===========================================================================

def _hojas(node_id: int, n: int, linkage: np.ndarray) -> List[int]:
    """Devuelve los indices de hoja (0..n-1) que pertenecen a un nodo del linkage."""
    if node_id < n:
        return [node_id]
    row = linkage[node_id - n]
    a, b = int(row[0]), int(row[1])
    return _hojas(a, n, linkage) + _hojas(b, n, linkage)


# Indices cuyos valores de distancia pueden salir del rango [0, 1]
_INDICES_NO_ACOTADOS = {'fager', 'kulezynski', 'correlation', 'correlation-ratio',
                        'correlation_ratio', 'radio'}


def cortar_dendrograma(
    linkage: np.ndarray,
    labels: List[str],
    umbral_porcentaje: float,
    sim_index: str = 'jaccard',
) -> pd.DataFrame:
    """
    Corta el dendrograma a un umbral de similitud y asigna cada estado a un
    grupo biogeografico (color).

    Funciona con TODOS los indices implementados:
      - Indices acotados [0,1]: jaccard, simpson, sorensen, ochiai,
                                braun-blanquet, baroni
        El umbral se interpreta directamente como porcentaje de similitud:
        umbral_distancia = 1 - (umbral_porcentaje / 100)
      - Indices NO acotados: fager, kulezynski, correlation
        Las distancias pueden ser negativas o > 1, por lo que el umbral se
        interpreta como percentil del rango real de distancias del linkage:
        umbral_distancia = dist_min + (umbral_porcentaje/100) * (dist_max - dist_min)
        Se emite una advertencia explicando esta conversion.

    Parametros
    ----------
    linkage : np.ndarray
        Matriz de linkage [cluster_a, cluster_b, distancia, tamano].
        La columna 'distancia' siempre es 1 - similitud.
    labels : List[str]
        Nombres de los estados en el mismo orden que los indices 0..n-1.
    umbral_porcentaje : float
        Umbral expresado como porcentaje entero (0-100).
        Para indices acotados: porcentaje de similitud (ej. 40 = similitud 0.40).
        Para indices no acotados: percentil del rango de distancias observado.
    sim_index : str
        Nombre del indice de similitud usado para generar el linkage.
        Necesario para detectar indices no acotados. Por defecto 'jaccard'.

    Retorna
    -------
    pd.DataFrame con columnas ['estado', 'color']
        'estado' : nombre del estado (str)
        'color'  : numero de grupo al que pertenece (int, empieza en 1)
    """
    if linkage is None or len(linkage) == 0:
        raise ValueError("El linkage esta vacio. Ejecuta primero el pipeline.")
    if not (0.0 <= umbral_porcentaje <= 100.0):
        raise ValueError(f"umbral_porcentaje debe estar entre 0 y 100, recibido: {umbral_porcentaje}")

    distancias = linkage[:, 2].astype(float)
    # Ignorar NaN que pueden aparecer con indices no acotados
    distancias_validas = distancias[~np.isnan(distancias)]

    indice_lower = sim_index.lower().strip()
    es_no_acotado = indice_lower in _INDICES_NO_ACOTADOS

    if es_no_acotado:
        # Para indices no acotados el umbral es un percentil del rango real
        dist_min = float(np.nanmin(distancias_validas))
        dist_max = float(np.nanmax(distancias_validas))
        umbral_distancia = dist_min + (umbral_porcentaje / 100.0) * (dist_max - dist_min)
        logger.warning(
            f"Indice '{sim_index}' no acotado a [0,1]: el umbral {umbral_porcentaje}% "
            f"se interpreta como percentil del rango de distancias observado "
            f"[{dist_min:.4f}, {dist_max:.4f}] → umbral_distancia = {umbral_distancia:.4f}. "
            f"Considera usar jaccard, sorensen u ochiai para umbrales en % de similitud directa."
        )
    else:
        # Para indices acotados: distancia = 1 - similitud
        umbral_distancia = 1.0 - (umbral_porcentaje / 100.0)
        logger.info(
            f"Indice '{sim_index}' acotado [0,1]: "
            f"umbral {umbral_porcentaje}% similitud → distancia <= {umbral_distancia:.4f}"
        )

        dist_min_real = float(np.nanmin(distancias_validas))
        dist_max_real = float(np.nanmax(distancias_validas))
        rango_real    = dist_max_real - dist_min_real

        # --- Trampa: umbral fuera del rango real de distancias ---------------
        # Si umbral_distancia >= dist_max_real, TODAS las fusiones quedan por
        # debajo del umbral → resultado trivial: un solo grupo.
        if umbral_distancia >= dist_max_real:
            sim_max_real = round((1.0 - dist_min_real) * 100, 1)
            sim_min_real = round((1.0 - dist_max_real) * 100, 1)
            logger.warning(
                f"ATENCION [{sim_index}]: el umbral {umbral_porcentaje}% de similitud "
                f"(distancia <= {umbral_distancia:.4f}) esta POR ENCIMA de la distancia "
                f"maxima real del linkage ({dist_max_real:.4f}). "
                f"Todas las fusiones quedaran incluidas → resultado: 1 solo grupo. "
                f"El rango real de similitudes en este linkage es "
                f"[{sim_min_real}%, {sim_max_real}%]. "
                f"Prueba un umbral entre {sim_min_real}% y {sim_max_real}%."
            )

        # --- Fantasma: rango de distancias muy comprimido --------------------
        # Ocurre tipicamente con Baroni cuando las ausencias compartidas son
        # muchas (dataset grande), inflando artificialmente la similitud.
        # Umbral arbitrario: si el rango real es menor a 0.20 (similitudes
        # todas por encima del 80%), el corte porcentual es poco discriminativo.
        UMBRAL_RANGO_ESTRECHO = 0.20
        if rango_real < UMBRAL_RANGO_ESTRECHO:
            sim_min_real = round((1.0 - dist_max_real) * 100, 1)
            sim_max_real = round((1.0 - dist_min_real) * 100, 1)
            logger.warning(
                f"ATENCION [{sim_index}]: el rango de distancias del linkage es muy "
                f"estrecho ({dist_min_real:.4f} a {dist_max_real:.4f}, rango={rango_real:.4f}). "
                f"Esto significa que todas las similitudes reales estan entre "
                f"{sim_min_real}% y {sim_max_real}%, probablemente por ausencias "
                f"compartidas que inflan el indice (tipico de Baroni con datasets grandes). "
                f"Un corte al {umbral_porcentaje}% puede producir muy pocos grupos o uno solo. "
                f"Considera: (a) subir el umbral a >{sim_min_real}%, "
                f"(b) usar un indice que ignore ausencias (jaccard, sorensen, ochiai)."
            )

    n = len(labels)

    # Union-Find con compresion de camino
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for row in linkage:
        a, b, dist = int(row[0]), int(row[1]), float(row[2])
        if np.isnan(dist):
            # Fusion con distancia indefinida: no unir (quedan separados)
            continue
        miembros_a = _hojas(a, n, linkage)
        miembros_b = _hojas(b, n, linkage)
        if dist <= umbral_distancia:
            for ha in miembros_a:
                for hb in miembros_b:
                    union(ha, hb)

    raices: Dict[int, int] = {}
    grupo_num = 0
    grupos = []
    for i in range(n):
        r = find(i)
        if r not in raices:
            grupo_num += 1
            raices[r] = grupo_num
        grupos.append(raices[r])

    resultado = pd.DataFrame({'estado': labels, 'color': grupos})
    resultado = resultado.sort_values('color').reset_index(drop=True)

    logger.info(
        f"Corte al {umbral_porcentaje}% → {grupo_num} grupo(s) formado(s) "
        f"(distancia <= {umbral_distancia:.4f})"
    )
    return resultado


# ===========================================================================
#  Visualizacion del dendrograma
# ===========================================================================

def _formato_dendrograma(ax, index_name: str, method: str) -> None:
    """Aplica etiquetas y titulo estandar al eje del dendrograma."""
    nombres = {
        'single':   'Single Linkage',
        'complete': 'Complete Linkage',
        'average':  'UPGMA (Average Linkage)',
    }
    ax.set_ylabel(f'Distancia (1 - {index_name})', fontsize=12)
    ax.set_xlabel('Estados', fontsize=12)
    ax.set_title(
        f'Dendrograma - {nombres.get(method, method)}\n(Indice: {index_name})',
        fontsize=14,
        fontweight='bold',
    )


def plot_dendrogram(
    linkage: np.ndarray,
    labels: List[str],
    png_path: Path,
    index_name: str = 'Jaccard',
    method: str = 'average',
    cthreshold: float | None = None,
) -> None:
    """Genera el dendrograma usando scipy si esta disponible; si no, usa implementacion propia.

    cthreshold : float | None
        Distancia de corte para colorear las ramas del dendrograma.
        Si se pasa, las ramas con distancia < cthreshold se colorean del
        mismo color que su grupo en el CSV de corte, logrando consistencia
        visual entre dendrograma y mapa.
        Si es None, scipy usa su default (70% de la altura maxima).
    """
    try:
        from scipy.cluster.hierarchy import dendrogram
        logger.info("scipy disponible: usando scipy.dendrogram.")
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 8))

        dendrogram(
            linkage,
            labels=labels,
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=10,
            color_threshold=cthreshold,   # sincroniza colores con el corte real
        )

        # Linea horizontal roja en el punto de corte
        if cthreshold is not None:
            ax.axhline(
                y=cthreshold,
                color='red',
                linestyle='--',
                linewidth=1.2,
                label=f'Corte: {cthreshold:.3f} (sim. {(1-cthreshold)*100:.1f}%)',
            )
            ax.legend(fontsize=10)

        _formato_dendrograma(ax, index_name, method)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Dendrograma guardado: {png_path}")
    except Exception:
        logger.warning("scipy no disponible; usando implementacion basica.")
        plot_dendrogram_simple(linkage, labels, png_path, index_name, method, cthreshold)


def plot_dendrogram_simple(
    linkage: np.ndarray,
    labels: List[str],
    png_path: Path,
    index_name: str = 'Jaccard',
    method: str = 'average',
    cthreshold: float | None = None,
) -> None:
    """Dibuja el dendrograma con matplotlib puro (fallback sin scipy)."""
    logger.info("Generando dendrograma basico (sin scipy)...")
    n = len(labels)
    if linkage.size == 0:
        logger.warning("No hay linkage para graficar.")
        return

    rows = linkage[:, :4].astype(float).tolist()

    node_members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    node_height: Dict[int, float] = {i: 0.0 for i in range(n)}
    x_pos: Dict[int, float] = {i: float(i) for i in range(n)}

    for k, row in enumerate(rows):
        a, b = int(row[0]), int(row[1])
        new_id = n + k
        node_members[new_id] = node_members.get(a, []) + node_members.get(b, [])
        node_height[new_id] = float(row[2])

    for k in range(len(rows)):
        new_id = n + k
        leaves = node_members[new_id]
        x_pos[new_id] = float(np.mean([x_pos[m] for m in leaves]))

    fig, ax = plt.subplots(figsize=(max(10, n * 0.5), 8))
    for k, row in enumerate(rows):
        a, b = int(row[0]), int(row[1])
        new_id = n + k
        xa, xb = x_pos[a], x_pos[b]
        ya, yb = node_height[a], node_height[b]
        yn = node_height[new_id]
        ax.plot([xa, xa], [ya, yn], lw=1.5, solid_capstyle='butt', color='steelblue')
        ax.plot([xb, xb], [yb, yn], lw=1.5, solid_capstyle='butt', color='steelblue')
        ax.plot([xa, xb], [yn, yn], lw=1.5, color='steelblue')

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha='right')
    if cthreshold is not None:
        ax.axhline(
            y=cthreshold,
            color='red',
            linestyle='--',
            linewidth=1.2,
            label=f'Corte: {cthreshold:.3f} (sim. {(1-cthreshold)*100:.1f}%)',
        )
        ax.legend(fontsize=10)

    _formato_dendrograma(ax, index_name, method)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Dendrograma basico guardado: {png_path}")


# ===========================================================================
#  Guardado de resultados
# ===========================================================================

def save_outputs(
    outdir: Path,
    presence_absence: pd.DataFrame,
    similarity: pd.DataFrame,
    linkage: np.ndarray,
    clusters: Dict[int, Dict[str, Any]],
    labels: List[str],
    index_name: str = 'jaccard',
    method: str = 'average',
    export_distance: bool = False,
    percent: bool = False,
    umbral_corte: float | None = None,
) -> None:
    """Guarda todos los CSVs y el dendrograma en el directorio de salida.

    umbral_corte : float | None
        Porcentaje de similitud del corte (ej. 25.0).
        Si se pasa, se calcula el color_threshold para que los colores del
        dendrograma sean consistentes con los grupos del CSV de corte.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    enc = 'utf-8-sig'  # compatibilidad con Excel en Windows (acentos)

    presence_absence.to_csv(outdir / 'presence_absence_matrix.csv', index=True, encoding=enc)
    logger.info("Guardado: presence_absence_matrix.csv")

    if not similarity.empty:
        similarity.to_csv(outdir / f'{index_name}_similarity_states.csv', index=True, encoding=enc)
        logger.info(f"Guardado: {index_name}_similarity_states.csv")

    if export_distance and not similarity.empty:
        sim_01 = similarity / 100.0 if percent else similarity
        (1.0 - sim_01).to_csv(
            outdir / f'{index_name}_distance_matrix_states.csv', index=True, encoding=enc
        )
        logger.info(f"Guardado: {index_name}_distance_matrix_states.csv")

    if linkage.size:
        pd.DataFrame(
            linkage, columns=['cluster_a', 'cluster_b', 'distance', 'size']
        ).to_csv(outdir / f'linkage_{method}.csv', index=False, encoding=enc)
        logger.info(f"Guardado: linkage_{method}.csv")

        # Calcular color_threshold para sincronizar colores del dendrograma
        # con los grupos del CSV de corte
        cthreshold = None
        if umbral_corte is not None:
            indice_lower = index_name.lower().strip()
            if indice_lower in _INDICES_NO_ACOTADOS:
                # Indices no acotados: usar percentil del rango real
                dists = linkage[:, 2].astype(float)
                dists_validas = dists[~np.isnan(dists)]
                if len(dists_validas) > 0:
                    d_min = float(np.nanmin(dists_validas))
                    d_max = float(np.nanmax(dists_validas))
                    cthreshold = d_min + (umbral_corte / 100.0) * (d_max - d_min)
            else:
                # Indices acotados: conversion directa
                cthreshold = 1.0 - (umbral_corte / 100.0)

        plot_dendrogram(
            linkage, labels,
            outdir / f'dendrogram_{method}.png',
            index_name.capitalize(), method,
            cthreshold=cthreshold,
        )
    else:
        logger.info("No hay linkage para guardar (muy pocos estados).")


# ===========================================================================
#  Utilidades
# ===========================================================================

def list_sheets(excel_path: Path) -> List[str]:
    """Lista todas las hojas disponibles en un archivo Excel."""
    try:
        return pd.ExcelFile(excel_path).sheet_names
    except Exception as e:
        logger.error(f"Error al leer el archivo Excel: {e}")
        return []


# ===========================================================================
#  Pipeline principal
# ===========================================================================

def pipeline(
    input_path: str,
    sheet: str | None,
    species_col: str | None,
    state_col: str | None,
    outdir: str,
    min_species: int = 0,
    percent: bool = False,
    export_distance: bool = False,
    sim_index: str = 'jaccard',
    linkage_method: str = 'average',
    umbral_corte: float | None = None,
) -> Dict[str, Any]:
    """Ejecucion completa del pipeline UPGMA.

    umbral_corte : float | None
        Si se especifica, se usa para sincronizar el color_threshold del
        dendrograma con el corte real. No ejecuta cortar_dendrograma —
        solo afecta los colores del PNG.
    """
    input_path = Path(input_path)
    outdir_path = Path(outdir)

    df = read_input_excel(input_path, sheet)
    species_col, state_col = detect_columns(df, species_col, state_col)
    pa = build_presence_absence(df, species_col, state_col)

    state_counts = pa.sum(axis=0)
    keep = state_counts[state_counts >= min_species].index.tolist()
    pa = pa.loc[:, keep]
    logger.info(f"Estados tras filtro (min_species={min_species}): {len(keep)}")

    if pa.shape[1] == 0:
        logger.warning("No quedan estados tras el filtrado. Terminando.")
        save_outputs(outdir_path, pa, pd.DataFrame(), np.array([]), {}, [],
                     sim_index, linkage_method, export_distance, percent,
                     umbral_corte=umbral_corte)
        return {'pa': pa, 'similarity': pd.DataFrame(), 'linkage': np.array([]), 'clusters': {}}

    sim = compute_similarity(pa, index_name=sim_index, percent=percent)

    sim_01 = sim.fillna(0.0)
    if percent:
        sim_01 = sim_01 / 100.0

    logger.info(f"Ejecutando clustering {linkage_method.upper()} con indice {sim_index.upper()}...")
    linkage, clusters, labels = hierarchical_linkage_from_similarity(sim_01, method=linkage_method)

    save_outputs(outdir_path, pa, sim, linkage, clusters, labels,
                 sim_index, linkage_method, export_distance, percent,
                 umbral_corte=umbral_corte)

    return {
        'pa': pa,
        'similarity': sim,
        'linkage': linkage,
        'clusters': clusters,
        'labels': labels,
        'index': sim_index,
        'method': linkage_method,
    }


# ===========================================================================
#  CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Pipeline de clustering jerarquico — multiples indices de similitud',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python upgma_compacto_modular.py -i datos.xlsx -o resultados\n"
            "  python upgma_compacto_modular.py -i datos.xlsx --sim-index baroni "
            "--umbral-corte 40 -o resultados\n"
            "  python upgma_compacto_modular.py -i datos.xlsx --list-sheets\n"
        ),
    )
    p.add_argument('--input', '-i', required=True,
                   help='Archivo Excel de entrada (.xlsx / .xls)')
    p.add_argument('--sheet', default=None,
                   help='Nombre o numero de hoja (por defecto: primera hoja)')
    p.add_argument('--species', default=None,
                   help='Nombre exacto de la columna de especie (autodeteccion si se omite)')
    p.add_argument('--state', default=None,
                   help='Nombre exacto de la columna de estado (autodeteccion si se omite)')
    p.add_argument('--out', '-o', default='out',
                   help='Directorio de salida (por defecto: out/)')
    p.add_argument('--min-species', type=int, default=0,
                   help='Excluir estados con menos de N especies (por defecto: 0)')
    p.add_argument('--percent', action='store_true',
                   help='Guardar similitud en 0-100 en lugar de 0-1')
    p.add_argument('--export-distance', action='store_true',
                   help='Exportar tambien la matriz de distancia (1 - similitud)')
    p.add_argument('--sim-index', default='jaccard',
                   help=(
                       'Indice de similitud (por defecto: jaccard). '
                       'Opciones: jaccard, simpson, sorensen, ochiai, braun-blanquet, '
                       'fager, kulezynski, correlation, baroni'
                   ))
    p.add_argument('--linkage-method', default='average',
                   help='Metodo de linkage (por defecto: average). Opciones: single, complete, average')
    p.add_argument('--umbral-corte', type=float, default=None, metavar='N',
                   help=(
                       'Corta el dendrograma al N%% de similitud y guarda '
                       'grupos_corte_{N}pct.csv. Ejemplo: --umbral-corte 40'
                   ))
    p.add_argument('--list-sheets', action='store_true',
                   help='Listar hojas del archivo Excel y salir')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # -- Listar hojas y salir ------------------------------------------------
    if args.list_sheets:
        sheets = list_sheets(Path(args.input))
        if sheets:
            print(f"\n Hojas disponibles en '{args.input}':")
            for i, s in enumerate(sheets):
                print(f"  {i}: {s}")
            print(f"\nUsa --sheet con el nombre exacto o el indice (0-{len(sheets) - 1})")
        else:
            print(f"No se pudieron leer las hojas de '{args.input}'")
        raise SystemExit(0)

    # -- Pipeline principal --------------------------------------------------
    resultados = pipeline(
        args.input,
        args.sheet,
        args.species,
        args.state,
        args.out,
        min_species=args.min_species,
        percent=args.percent,
        export_distance=args.export_distance,
        sim_index=args.sim_index,
        linkage_method=args.linkage_method,
        umbral_corte=args.umbral_corte,
    )

    # -- Corte del dendrograma (opcional) ------------------------------------
    if args.umbral_corte is not None:
        linkage = resultados.get('linkage')
        labels  = resultados.get('labels')

        if linkage is not None and len(linkage) > 0 and labels:
            umbral = args.umbral_corte
            grupos = cortar_dendrograma(linkage, labels, umbral, sim_index=args.sim_index)

            outdir = Path(args.out)
            nombre_csv = outdir / f'grupos_corte_{int(umbral)}pct.csv'
            grupos.to_csv(nombre_csv, index=False, encoding='utf-8-sig')
            logger.info(f"Grupos guardados en: {nombre_csv}")

            print(f"\n{'─' * 55}")
            print(f"  Corte al {umbral}% de similitud → {grupos['color'].nunique()} grupos")
            print(f"{'─' * 55}")
            for g_id, sub in grupos.groupby('color'):
                estados = ', '.join(sub['estado'].tolist())
                print(f"  Grupo {g_id:>2}: {estados}")
            print(f"{'─' * 55}\n")
        else:
            logger.warning("No se pudo aplicar el corte: linkage vacio o sin etiquetas.")
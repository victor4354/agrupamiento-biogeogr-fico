# Biogeographic Clustering Pipeline (UPGMA)

Hierarchical clustering pipeline for biogeographic analysis of species distribution data. Developed during a one-year research collaboration with UNIBIO (Instituto de Biología, UNAM).

## Overview

- Processes presence/absence data of **3,030 Asteraceae species** across **32 Mexican states** (14,113 records).
- Implements **9 similarity indices** from scratch with vectorized NumPy operations.
- Supports **3 linkage methods** (single, complete, UPGMA) with custom hierarchical clustering and Union-Find for dendrogram cutting.
- Robust input validation, automatic column detection, and graceful fallback (scipy → pure matplotlib).

## Stack

Python 3.10+, NumPy, pandas, matplotlib, scipy (optional), openpyxl.

## Project structure

```
.
├── upgma_compacto_modular.py   # Core pipeline (similarity, clustering, viz)
├── configuracion_upgma.py      # Parameter configuration + validation
├── datos.xlsx                  # Input dataset (species × state records)
├── requirements.txt
└── README.md
```

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with default config

```bash
python configuracion_upgma.py
```

### 3. Or use the CLI directly

```bash
python upgma_compacto_modular.py \
    -i datos.xlsx \
    -o resultados \
    --sim-index jaccard \
    --linkage-method average \
    --umbral-corte 40
```

## Similarity indices implemented

| Index            | Formula                              | Range  |
|------------------|--------------------------------------|--------|
| Jaccard          | \|A ∩ B\| / \|A ∪ B\|                | [0, 1] |
| Simpson          | \|A ∩ B\| / min(\|A\|, \|B\|)        | [0, 1] |
| Sørensen-Dice    | 2·\|A ∩ B\| / (\|A\| + \|B\|)        | [0, 1] |
| Ochiai           | \|A ∩ B\| / √(\|A\|·\|B\|)           | [0, 1] |
| Braun-Blanquet   | \|A ∩ B\| / max(\|A\|, \|B\|)        | [0, 1] |
| Baroni-Urbani-Buser | accounts for shared absences      | [0, 1] |
| Fager            | unbounded variant                    | ≈      |
| Kulezynski       | unbounded variant                    | ≈      |
| Correlation ratio| unbounded variant                    | ≈      |

For unbounded indices, the cut threshold is interpreted as a percentile of the observed distance range (with explicit warnings).

## Outputs

The pipeline writes to the configured output directory:

- `presence_absence_matrix.csv` — species × states binary matrix.
- `{index}_similarity_states.csv` — pairwise similarity between states.
- `{index}_distance_matrix_states.csv` — distance matrix (1 − similarity).
- `linkage_{method}.csv` — linkage table compatible with scipy format.
- `dendrogram_{method}.png` — dendrogram (with optional cut line).
- `grupos_corte_{N}pct.csv` — group assignment at the configured threshold.

## Design notes

- **Vectorized similarity computation**: all 9 indices are computed via NumPy matrix operations, no Python-level loops over species pairs.
- **Custom clustering**: produces a scipy-compatible linkage matrix without requiring scipy at compute time. Scipy is only used for prettier dendrograms when available.
- **Edge-case detection**: the pipeline emits warnings when (a) the cut threshold falls outside the actual distance range of the linkage, or (b) the distance range is too narrow (typical of Baroni with large datasets where shared absences inflate similarity).

## Context

Developed as part of a year-long social service (servicio social) at UNIBIO, Instituto de Biología, UNAM, supervising taxonomist-led biogeographic analysis of the Asteraceae family in Mexico.

## Author

**Victor** — Data Scientist
[LinkedIn](https://www.linkedin.com/in/v%C3%ADctor-becerra-jim%C3%A9nez-184356407/) · [GitHub](https://github.com/victor4354/agrupamiento-biogeogr-fico) · becerra.jimenez.victor1@gmail.com


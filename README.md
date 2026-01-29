# FedAugment

![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Flbhm%2Ffedaugment%2Fmain%2Fpyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/lbhm/fedaugment)](https://github.com/lnhm/fedaugment/blob/main/LICENSE)

This repository contains the source code, experiment logs, and result analyses for our paper
"FedAugment: Table Augmentation Search over Decentralized Data Repositories".

## 🏗️ Architecture Overview

The repository is structured as follows:

```bash
fedaugment/
├── analysis        # Jupyter notebooks with result analyses and plotting code
├── experiments     # Python and Bash scripts with experiment configurations
├── logs            # results of our experimental evaluation
├── scripts         # utility scripts for data processing and evaluation
├── src/fedaugment  # main Python package with our implementation
└── tests           # unit tests for part of the codebase
```

The following diagram illustrates the overall architecture of the FedAugment workflow:

```text
+----------------------+
|    Raw Table Data    |
| (CSV/Parquet files)  |
+----------+-----------+
           |
           v
+----------------------------------------------------------------------------------+
|                             1. EMBEDDING GENERATION                              |
|     +-------------+   +-------------+   +-------------+       +-------------+    |
|     |  View 1     |   |  View 2     |   |  View 3     |  ...  |  View N     |    |
|     | mpnet +     |   | distilrob + |   | gte_base +  |       | qwen3_8b +  |    |
|     | dj_adpt     |   | dj_adpt     |   | dj_adpt     |       | dj_adpt     |    |
|     +------+------+   +------+------+   +------+------+       +------+------+    |
|            |                 |                 |                     |           |
|            v                 v                 v                     v           |
|        [384-dim]         [768-dim]         [768-dim]             [4096-dim]      |
|        embeddings        embeddings        embeddings            embeddings      |
+------------+-----------------+-----------------+---------------------+----------+
             |                 |                 |                     |
             +-----------------+----------+------+---------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                           2. PROJECTION MODEL TRAINING                           |
|                                                                                  |
|  Training Data                Projection Models:                                 |
|  +---------------------+      - CL (Contrastive Learning) --- Neural network     |
|  | Curated subset      |      - LA2M (Local Isometry) ------- Clustering-based   |
|  | (FFT/Grid/Random)   |      - Vec2Vec --------------------- GAN-based          |
|  +---------------------+      - Procrustes ------------------ Orthogonal align   |
|                                                                                  |
|  Output: Learned transformations that map all views to a common vector space     |
+-----------------------------------------+----------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                            3. ALIGNED EMBEDDING SPACE                            |
|                                                                                  |
|      View 1        View 2        View 3     ...      View N                      |
|        |             |             |                   |                         |
|        +-------------+-------------+-------------------+                         |
|                              |                                                   |
|                    +---------v---------+                                         |
|                    |      Common       |                                         |
|                    |  Embedding Space  |                                         |
|                    +---------+---------+                                         |
|                              |                                                   |
|                    +---------v---------+                                         |
|                    |    HNSW Index     |  < Fast approximate nearest neighbor    |
|                    +-------------------+                                         |
+-----------------------------------------+----------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                            4. TABLE AUGMENTATION TASKS                           |
|                                                                                  |
|        +-----------------------------+    +-----------------------------+        |
|        |       JOIN DISCOVERY        |    |       UNION DISCOVERY       |        |
|        |                             |    |                             |        |
|        |  Query: Column A            |    |  Query: Table X             |        |
|        |     v                       |    |     v                       |        |
|        |  Find columns that can      |    |  Find tables with           |        |
|        |  be joined with A           |    |  compatible schemas         |        |
|        |     v                       |    |     v                       |        |
|        |  Metrics: P@k, R@k, MAP     |    |  Metrics: P@k, R@k, MAP     |        |
|        +-----------------------------+    +-----------------------------+        |
+----------------------------------------------------------------------------------+
```

## 🚀 Getting Started

We recommend using [uv](https://docs.astral.sh/uv/) to manage `fedaugment` and its dependencies.
Installing the project is as simple as:

```bash
uv sync
```

We offer the following optional dependency groups (use `uv sync --extra <group>`):

- PyTorch variants (mutually exclusive):
  - `cpu`: Force a CPU installation of PyTorch
  - `cu126`: Force a CUDA 12.6 installation of PyTorch
  - `cu128`: Force a CUDA 12.8 installation of PyTorch
- `experiments`: Installs additional dependencies for running experiments
- `flash-attn`: For more efficient attention operators in PyTorch

Note that the default PyTorch version depends on your operating system (CPU for Windows and
Mac, CUDA 12.x for Linux).

## 📂 Datasets

### Overview

We use the following datasets in our experiments:

- [Freyja](https://mydisk.cs.upc.edu/s/QHJbKcyeacxq35f)
- [OmniMatch City (Test set)](https://zenodo.org/records/15705578)
- [OmniMatch Culture (Test set)](https://zenodo.org/records/15705578)
- [Santos (Small)](https://zenodo.org/records/7758091)
- [LakeBench (WebTable)](https://github.com/DB-121143/LakeBench?tab=readme-ov-file#prepare-datasets)

### Dataset Folder Structure

The project expects datasets in the following structure (symlinked or stored at `data/`):

```bash
data/
│
├── datasets/                              # Raw tabular data
│   │
│   └── {dataset_name}/                    # e.g., webtable, omnimatch_city_train, santos_small, freyja
│       │
│       ├── datasets/                      # Full original dataset
│       │   ├── pq/                        # Parquet files
│       │   │   └── {table_id}.pq
│       │   └── csv/                       # CSV files
│       │       └── {table_id}.csv
│       │
│       ├── queries/                       # Query tables and ground truth (queries are optional, if they don't exist, we use tables from datasets/)
│       │   ├── pq/                        # Query tables (parquet)
│       │   ├── csv/                       # Query tables (csv)
│       │   ├── join_queries.csv           # Join query list (only for join tasks)
│       │   ├── join_ground_truth.csv      # Join ground truth pairs (only for join tasks)
│       │   ├── union_queries.csv          # Union query list (only for union tasks)
│       │   └── union_ground_truth.csv     # Union ground truth pairs (only for union tasks)
│       │
│       ├── split/                         # Train/test/val splits (webtable only)
│       │   ├── train/
│       │   ├── test/
│       │   └── val/
│       │
│       └── sample-{pct}/                  # Sampled subsets from data curation (webtable only)
│           └── {curation_method}/         # e.g., fft_cos-mpnet-dj_adpt-k=63049
│               └── {table_id}.pq
│
│
└── embeddings/                            # Generated embeddings
    │
    └── {dataset_name}/                    # Mirrors datasets/ structure
        │
        ├── datasets/                      # Dataset embeddings (full corpus)
        │   │
        │   └── {model}-{strategy}.fa/     # Feature archive per embedding pipeline
        │       ├── embeddings.npy         # (N, D) float32 array
        │       ├── column_ids.npy         # (N,) string array: "{table_id}::{column_name}"
        │       └── metadata.json          # Pipeline metadata
        │
        ├── queries/                       # Query embeddings (if there are no dedicated queries, we symlink to datasets/)
        │   │
        │   └── {model}-{strategy}.fa/     # Feature archive per embedding pipeline
        │       ├── embeddings.npy         # (N, D) float32 array
        │       ├── column_ids.npy         # (N,) string array: "{table_id}::{column_name}"
        │       └── metadata.json          # Pipeline metadata
        │
        ├── sample-{pct}/                  # Embeddings for sampled subsets (percentages: 001, 005, 050)
        │   │
        │   └── {curation_method}/         # Curation strategy
        │       └── {model}-{strategy}.fa/
        │           ├── embeddings.npy
        │           ├── column_ids.npy
        │           └── metadata.json
        │
        └── split/                         # Embeddings for splits (webtable only)
            ├── train/
            │   └── {model}-{strategy}.fa/
            ├── test/
            │   └── {model}-{strategy}.fa/
            └── val/
                └── {model}-{strategy}.fa/
```

### File Formats

**Query Files**:

- `join_queries.csv`: List of join query columns

  ```csv
  query_table,query_column
  csvData1549285__2.csv,AST%
  csvData1549285__2.csv,BLK%
  ```

- `join_ground_truth.csv`: Ground truth join pairs

  ```csv
  query_table,candidate_table,query_column,candidate_column
  csvData1549285__2.csv,csvData20409520__4.csv,DRtg,DRtg
  ```

**Embedding Files** (stored in `.fa/` directories):

- `embeddings.npy`: NumPy array of shape `(N_columns, embedding_dim)`, dtype `float32`
- `column_ids.npy`: NumPy array of shape `(N_columns,)`, dtype `StringDType()`, format `"{table_id}::{column_name}"`
- `metadata.json`: Pipeline metadata including model, shape, and source information

### File Naming Conventions

- **Dataset names**: `webtable`, `omnimatch_city_test`, `omnimatch_culture_test`, `santos_small`, `freyja`
- **Embedding pipelines**: `{model}-{strategy}`
  - Models: `mpnet`, `distilroberta`, `gte_base`, `gtr_t5`, `jina_v3`, `mini_l12`, `mini_l6`, etc.
  - Strategies: `dj_orig` (DeepJoin original), `dj_adpt` (DeepJoin adapted), etc.
- **Curation methods**: `{algorithm}[_{metric}]-{model}-{strategy}-k={n_columns}[-pca={variance}]`
  - Algorithms: `fft`, `grid`, `random`
  - Metrics: `cos` (cosine), `euc` (Euclidean)
  - Example: `fft_cos-mpnet-dj_adpt-k=63049-pca=0.9`

## 🧪 Experiments

To reproduce the experiments in our paper, first ensure you have downloaded the required datasets and placed them in the `data/datasets/` directory as described above.
If you prefer to store the datasets elsewhere, create a symbolic link named `data` in the project root pointing to your dataset directory. For example:

```bash
ln -s /your/local/storage data
```

After setting up the datasets, you can run all experiments using:

```bash
bash experiments/run_all.sh
```

### Hardware Requirements

- 500+ GB RAM
- 1.5+ TB disk space
- NVIDIA A100 80GB GPU or better

## 📖 Citation

TBD

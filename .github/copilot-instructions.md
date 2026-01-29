# FedAugment AI Agent Instructions

## Project Overview

FedAugment is a research project for **table augmentation search over decentralized data repositories**. The core problem: different embedding models create incompatible vector representations, making it impossible to merge/search across heterogeneous embedding databases. FedAugment trains projection models to map embeddings from different sources into a common space while preserving semantic similarity.

## Architecture

### Four-Pipeline Flow

1. **Data Curation** (`src/fedaugment/curation/`)

- `CurationManager`: Generates proxy column embeddings for a table collection (via `EmbeddingModel` + `PromptStrategy`) and writes curated column projections to disk.
- `Curator` interface + implementations (`FarthestFirstTraversal`, `KMedoids`, `GridSampling`, `RandomSampling`): Select representative columns by returning a subset of indices of the embedding matrix.

2. **Embedding Generation** (`src/fedaugment/embeddings/`)

- `PipelineComposer`: Orchestrates embedding generation across model×strategy combinations
- `EmbeddingModel`: Base class for models (OpenAI, SentenceTransformers)
- `PromptStrategy`: Defines how table data become text prompts (cell-level, column-level, frequency-ranked)
- Output: `.npy` files in `data/embeddings/` organized by pipeline name
  - Each pipeline output consists of a folder with a `.fa` suffix and a custom structure documented in `NPYDataset`.

3. **Projection Training** (`src/fedaugment/projections/`)

- Trains models to align heterogeneous embeddings into a common space
- Five model types: `ContrastiveLearningModel`, `NaiveModel`, `LocalIsometryModel`, `ProcrustesModel`, `Vec2VecModel`
  - `ContrastiveLearningModel` is our contribution, the others are baselines
- Dataset class: `NPYDataset` takes folder paths with embeddings from individual embeddings pipelines/views as input and collates all embeddings/views from the same underlying table column as one dataset item.
- Input: Embedding files from step 2
- Output: Model checkpoints and training logs

4. **Augmentation Search** (`src/fedaugment/augmentations/`)

- Purpose: Evaluate table-augmentation predictions on benchmark datasets.
- Key classes:
  - `JoinDiscoveryEvaluator`: computes join predictions for each query column and computes retrieval metrics.
  - `UnionDiscoveryEvaluator`: computes union predictions for each query table using weighted bipartite graph matching and computes retrieval metrics.
- Inputs: Embedding collections, projection model checkpoint, search queries, ground-truth join/union pairs. Outputs: metrics per query in a data frame.
  - Expects `.npy` files that comply with the expected structure of `NPYDataset`.

### Data Flow

```
CSV/Parquet tables → PipelineComposer → Column embeddings → ProjectionModel → Common space
(data/datasets/)                        (data/embeddings/)  (logs/)
```

## Configuration System

**Everything is Pydantic-based** (`src/fedaugment/config.py`). Configs use discriminated unions if needed.

- `TrainingConfig`: Top-level experiment configuration
- `DataModuleConfig`: Dataset loading (uses `class_` field: "DefaultDataModule" | "SingleStepTrainingDataModule")
- `ProjectionModelConfig`: Model architecture (uses `class_` + discriminated `module_kwargs`)
- `CriterionConfig`, `OptimizerConfig`, `LRSchedulerConfig`: Training parameters
- `TrainerConfig`: PyTorch Lightning trainer settings

**Pattern**: Use `getattr(module, config.class_)` to dynamically instantiate classes from string names.

## Critical Development Patterns

### 1. Dataset Management

- Embeddings stored in `.npy` format for fast reads using memory-mapped files
- `NPYDataset` lazily opens files in worker processes (see `src/fedaugment/projections/dataset.py`)
- Folder structure of an exemplary embedding dataset:
  ```
  <dataset_name>.fa
  |- embeddings.npy
  |- column_ids.npy
  |- metadata.json
  ```

### 2. PyTorch Lightning Integration

- All projection models extend `L.LightningModule` via `ProjectionModel` base class
- Metrics use orchestrator pattern (`CosineSimilarityOrchestrator` + `OrchestratorChild`) to share computation
- Batched metric computation controlled by `metric_batch_size` to prevent OOM
  - Computing the entire similarity matrix at once exceeds GPU memory for large datasets

### 3. Multi-View Learning

- Batch shape: `(B, V, D)` where B=batch size, V=number of views/models, D=embedding dimension
- Each view represents the same column embedded by a different pipeline
- Losses expect this shape (see `MultiViewLoss` in `src/fedaugment/projections/losses.py`)

### 4. Experiment Setup

**Always call** `experiment_setup()` at script start (from `fedaugment.utils`):

- Configures `loguru` logging with `tqdm` integration
- Sets multiprocessing to `forkserver` mode
- Changes to the Git root directory
- Enables CUDA expandable segments for memory efficiency

## Running Code

### Dependencies

Install with `uv sync` (requires `uv` from astral.sh):

```bash
uv sync                     # Default (CPU on Win/Mac, CUDA on Linux)
uv sync --extra cu128       # Force CUDA 12.8
uv sync --extra experiments # Add Jupyter, plotting libs
```

### Typical Workflows

**Run tests**:

```bash
uv run pytest tests/
```

## Code Conventions

### Type Hints

- **Always use type hints** - project has `py.typed` and strict mypy config
- Custom types in `src/fedaugment/types.py`: `EmbeddingArray`, `ColIdArray`, `FloatArray`, `IntArray`, `StrPath`
- Use `type` statements for type aliases (Python 3.12+ feature)

### Ruff Configuration

- Line length: 99 characters
- Aggressive linting (`select = ["ALL"]`) with specific ignores
- Uses preview features + `skip-magic-trailing-comma`
- Google-style docstrings
- Experiments/scripts exempt from module docstring requirements

### Naming Patterns

- PEP 8 with extended ignore names: `B`, `D`, `F`, `L`, `N`, `V` (see pep8-naming config)
- Pipeline aliases: lowercase abbreviations (e.g., "mini_l6", "openai_3_small")
  - Separate model and strategy with hyphen ("-"), use underscores within names
- Strategy suffixes: "-col", "-cell", "-table"

### File Organization

- `experiments/`: Runnable scripts (not importable packages, INP001 ignored)
- `src/fedaugment/`: Importable library code
- `tests/`: pytest tests with shared fixtures in `conftest.py`

## Key Files to Reference

- `src/fedaugment/config.py`: All configuration schemas
- `src/fedaugment/projections/main.py`: Entry point for training
- `src/fedaugment/embeddings/pipeline_composer.py`: Embedding generation orchestration

## Common Gotchas

1. **Don't forget `experiment_setup()`** - Sets up logging, multiprocessing, CUDA config
2. **Column IDs verification** - Set `verify_column_ids=False` when peeking at dataset structure
3. **Data splits must sum to 1.0** - Validated by Pydantic
4. **Metrics orchestrator** - Share `CosineSimilarityOrchestrator` across metrics to avoid redundant similarity computation
5. **OpenAI API** - Requires `OPENAI_API_KEY` environment variable
6. **Test coverage** - The tests are not exhaustive yet; don't rely only on tests to understand functionality

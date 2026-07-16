# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install
```bash
pip install -e '.[all]' 'paddlex@git+https://github.com/PaddlePaddle/PaddleX.git@develop'
```

### Run Tests
```bash
pytest --verbose tests/
# Run a single test file
pytest --verbose tests/pipelines/test_ocr.py
# Run resource-intensive tests (excluded by default)
pytest --verbose -m resource_intensive tests/
```

### Code Style
```bash
pre-commit run --all-files
# Individual tools:
black paddleocr/
flake8 paddleocr/
```

## Architecture

PaddleOCR v3.x is a thin Python wrapper around **PaddleX** (Baidu's unified ML framework). All actual model inference and pipeline execution happens inside PaddleX — PaddleOCR provides the public API, CLI interface, and backward compatibility shims.

### Package Layout

- **`paddleocr/_pipelines/`** — Composite pipelines that chain multiple models:
  - `ocr.py`: Main `PaddleOCR` class (text detection + recognition + optional preprocessing)
  - `pp_structurev3.py`: `PPStructureV3` for layout-aware Markdown/JSON document conversion
  - `paddleocr_vl.py`: `PaddleOCRVL` wrapping the PaddleOCR-VL-1.5 (0.9B VLM)
  - `pp_chatocrv4_doc.py`, `doc_understanding.py`, `formula_recognition.py`, etc.
  - All inherit from `PaddleXPipelineWrapper` which delegates to `paddlex.create_pipeline()`

- **`paddleocr/_models/`** — Single-model wrappers (~19 models):
  - `text_detection.py`, `text_recognition.py`, `layout_detection.py`, `doc_vlm.py`, etc.
  - All inherit from `PaddleXPredictorWrapper` which delegates to `paddlex.create_predictor()`

- **`paddleocr/_cli.py`** — Registers all pipelines and models as CLI subcommands; entry point is `paddleocr/__main__.py`

- **`paddleocr/_utils/`** — Shared utilities: `cli.py` (argument parsing), `logging.py`, `deprecation.py`

- **`ppocr/`** — Legacy v2.x training code (data loaders, model definitions, losses, metrics). Not used by v3.x inference; kept for training workflows.

- **`ppstructure/`** — Legacy v2.x document structure code. Superseded by `PPStructureV3`.

- **`configs/`** — Training configs for legacy v2.x models, organized by task (`det/`, `rec/`, `cls/`, `table/`, `kie/`, `e2e/`).

### Data Flow (Inference)

```
User calls PaddleOCR().predict(image)
  → PaddleXPipelineWrapper._pipeline (PaddleX pipeline object)
    → PaddleX handles model loading, preprocessing, batching, postprocessing
  → Results returned as PaddleX result objects with .save_to_*() methods
```

### Test Structure

```
tests/
├── pipelines/       # Integration tests per pipeline class
├── models/          # Unit tests per model wrapper
├── test_*.py        # Legacy postprocess/utility tests
└── testing_utils.py # check_simple_inference_result(), check_wrapper_simple_inference_param_forwarding()
```

Heavy tests are marked `@pytest.mark.resource_intensive` and excluded from default runs via `pyproject.toml` (`addopts = "-m 'not resource_intensive'"`).

## Key Notes

- When adding a new model or pipeline, register it in `paddleocr/_cli.py` and export it from `paddleocr/__init__.py`.
- The `langchain-paddleocr/` subdirectory is excluded from pre-commit hooks.
- Python 3.8–3.13 compatibility is required; avoid 3.9+ syntax without guards.

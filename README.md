[![Tests](https://github.com/surbhisingh3359/AI-Business-Analytics-Assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/surbhisingh3359/AI-Business-Analytics-Assistant/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab.svg)](requirements.txt)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61-ff4b4b.svg)](https://streamlit.io)

# 📊 AI Business Analytics Assistant

_A dataset-agnostic analysis pipeline: profile → findings → decisions → bounded prediction._

Point it at any tabular file and it works out what the columns *mean* — identifiers,
metrics, dimensions, dates — then runs the analyses that schema supports, ranks the
findings by strength, and states the decision each finding can and cannot justify.
There are no hardcoded column names, so the same pipeline handles a sales extract, an
HR export, or a survey dump.

🖥️ Streamlit workspace · 🔌 [FastAPI service](api.py) · 🐍 [Python API](src/business_analysis_pipeline.py)

> [!IMPORTANT]
> **The analysis layer is deterministic, not generative.** Every insight, ranking,
> decision, and metric is computed from the data in `src/`, and each one carries the
> `evidence` dictionary it was derived from — so the same input always produces the
> same output and every claim is traceable to numbers. An LLM is **optional** and used
> only to narrate results ([`src/analysis_service.py`](src/analysis_service.py)); it
> never produces a figure. See [Architecture](#architecture) for where the boundary sits.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Compatibility](#compatibility)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [License](#license)

## Features

- **Infer** semantic column roles — identifier, metric, dimension, temporal — instead of
  assuming a fixed schema ([`src/dataset_classifier.py`](src/dataset_classifier.py)).
- **Audit** completeness, duplication, and constant columns before any analysis runs, and
  promote quality problems to first-class findings ([`src/data_quality.py`](src/data_quality.py)).
- **Analyse** across six families: group comparison, correlation, trend, risk, data
  quality, and categorical concentration ([`src/universal_analytics.py`](src/universal_analytics.py)).
- **Rank and de-duplicate** findings by priority, confidence, and type, so the top five
  are five different facts rather than five phrasings of one ([`src/insight_ranker.py`](src/insight_ranker.py)).
- **Ground** each finding in a decision object — the question it answers, the follow-up
  analysis it justifies, and the limitation that bounds it ([`src/decision_engine.py`](src/decision_engine.py)).
- **Predict** with a bounded baseline: auto-detect regression versus classification,
  score against a naive baseline, and decline to model when the data cannot support it
  ([`src/predictive_analysis.py`](src/predictive_analysis.py)).
- **Serve** one pipeline through three surfaces — Streamlit workspace, REST service, and
  a plain Python function.

## Installation

Python 3.11 and a virtual environment:

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

Then launch the workspace:

```bash
streamlit run app.py
```

## Configuration

Everything except optional local LLM narration runs with no configuration. To enable
narration, install and run [Ollama](https://ollama.com), copy `.env.example` to `.env`,
and select a local model:

| Variable | Purpose | Required |
| --- | --- | --- |
| `OLLAMA_MODEL` | Local model used by the optional `/ai-analysis` endpoint | No |

`.env` is gitignored; only the example template is tracked.

## Compatibility

| Component | Version | Notes |
| --- | --- | --- |
| Python | `3.11` | Tested against 3.11; typing uses `from __future__ import annotations` |
| pandas | `3.0.5` | Copy-on-write semantics assumed throughout |
| scikit-learn | `1.9.0` | `Pipeline` + `ColumnTransformer`; required for the predictive workspace |
| Streamlit | `1.61.1` | Uses `width="stretch"`, `st.html`, and theme sections in `.streamlit/config.toml` |
| Input formats | `.csv`, `.xlsx` | Supported by the Streamlit workspace and `/analyze`; `.xlsx` needs `openpyxl` |

## Usage Examples

### Analyse a file

```python
from src.business_analysis_pipeline import run_business_analysis_from_csv

result = run_business_analysis_from_csv(
    "data/raw/sample_sales.csv",
    max_insights=3,
    max_decisions=2,
)

print(result["status"])                    # 'completed'
print(result["insights"][0]["title"])      # 'Electronics has the highest average price'
print(result["insights"][0]["evidence"])   # the numbers behind the sentence
```

_→ Full API: [`src/business_analysis_pipeline.py`](src/business_analysis_pipeline.py)_

### Analyse a DataFrame in memory

```python
import pandas as pd
from src.business_analysis_pipeline import run_business_analysis

frame = pd.read_csv("data/raw/sample_sales.csv")
result = run_business_analysis(frame, max_insights=5, max_decisions=5)

for insight in result["insights"]:
    print(insight["priority"], "|", insight["title"])
```

_→ Full API: [`src/business_analysis_pipeline.py`](src/business_analysis_pipeline.py)_

### Read the decision layer

Each decision names its own limits, which is the point of the layer.

```python
for decision in result["decisions"]:
    print(decision["decision"])
    print("  justifies:", decision["recommended_analysis"])
    print("  confidence:", decision["confidence"])
    for limitation in decision["limitations"]:
        print("  limited by:", limitation)
```

_→ Full API: [`src/decision_engine.py`](src/decision_engine.py)_

### Train a bounded predictive baseline

```python
from src.predictive_analysis import (
    can_run_prediction,
    detect_prediction_targets,
    detect_prediction_task,
    run_prediction,
)

if can_run_prediction(frame):                      # False below 10 usable rows
    targets = detect_prediction_targets(frame)     # regression + classification candidates
    task = detect_prediction_task(frame, "revenue")

    prediction = run_prediction(frame, "revenue")
    print(prediction["model_name"])                # 'LinearRegression'
    print(prediction["metrics"])                   # {'mae': ..., 'rmse': ..., 'r2': ...}
    print(prediction["baseline_metrics"])          # the naive model it must beat
    print(prediction["model_quality"])             # whether it actually did
```

_→ Full guide: [`src/predictive_analysis.py`](src/predictive_analysis.py) — reports
`excluded_features` and `was_sampled` so a good score can be read in context._

### Serve it over HTTP

```bash
uvicorn api:app --reload
```

```bash
curl -F "file=@data/raw/sample_sales.csv" http://127.0.0.1:8000/analyze
```

_→ Full API: [`api.py`](api.py) — `/health`, `/analyze`, `/ai-analysis`_

## Architecture

The deterministic path never calls a model; the generative path only formats what the
deterministic path already computed.

```
                    CSV / XLSX / DataFrame
                              │
                     ┌────────┴────────┐
                     │  data_loader    │  load, clean, quality-audit
                     │  data_quality   │
                     └────────┬────────┘
                              │
                     ┌────────┴────────┐
                     │ dataset_        │  identifier / metric / dimension / temporal
                     │ classifier      │
                     └────────┬────────┘
                              │
                     ┌────────┴────────┐
                     │ analysis_       │  choose the analyses this schema supports
                     │ planner         │
                     └────────┬────────┘
                              │
                     ┌────────┴────────┐
                     │ universal_      │  group, correlate, trend, risk, concentration
                     │ analytics       │
                     └────────┬────────┘
                              │
                     ┌────────┴────────┐
                     │ insight_engine  │  findings + evidence
                     │ insight_ranker  │  score, de-duplicate, cap
                     └────────┬────────┘
                              │
                     ┌────────┴────────┐
                     │ decision_engine │  question · justified analysis · limitation
                     │ analysis_       │  run the follow-up analysis
                     │ execution       │
                     └────────┬────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   app.py            predictive_analysis     analysis_service
   (Streamlit)       (bounded baseline)      (optional LLM narration)
```

`app.py` is a presentation layer only — it imports `run_business_analysis` and
`run_prediction` and holds no analysis logic, so the pipeline is testable without a
browser. Its navigation is a CSS-magnified dock modelled on
[reactbits.dev/components/dock](https://reactbits.dev/components/dock); results are held
in session state, so switching views never recomputes anything.

## Testing

```bash
pytest -q
```

156 tests across 32 files, covering each pipeline stage plus the integration path from
DataFrame to executed decision. The suite needs no network, no API key, and no fixtures
beyond generated frames.

## Known Limitations

Stated deliberately — the pipeline refuses to guess where it cannot know.

- **Correlation is not causation, and the decision layer says so.** Every decision object
  carries a `limitations` list for exactly this reason.
- **The predictive layer is a baseline, not a tuned model.** Linear and logistic
  regression only, with no hyperparameter search; its job is to establish whether the
  target is predictable at all.
- **Time-series handling is trend detection, not forecasting.** No seasonality
  decomposition or autocorrelation modelling.
- **High-cardinality categoricals are dropped** above the one-hot ceiling rather than
  target-encoded; the excluded columns are reported instead of silently ignored.
- **`run_business_prediction` returns a reduced result.** It omits `baseline_metrics`
  and `model_quality`, so prefer `run_prediction` when you need to know whether the
  model beat its baseline.

## License

MIT — see [LICENSE](LICENSE).

# Drift Module

## Current Layout

`drift.demo`

Current offline demo pipeline. This is the main path used for the local monitoring demo.

- `phase_worker.py`: generates phased synthetic traffic and writes SQLite events.
- `metrics_worker.py`: computes drift/quality windows from SQLite and writes exporter artifacts.

`drift.store`

SQLite event store for demo prediction events.

`drift.replay`

Replay fallback for generated JSONL messages.

`drift.online`

Experimental online synthetic workers. These are not the primary accepted demo path yet.

`drift.monitoring`

Artifact-based Prometheus exporter.

`drift.synthetic_api`

FastAPI service for LLM generator/expert roles.

`drift.runner`

Legacy batch runner plus shared code. New code should not add orchestration here unless it is intentionally supporting the legacy runner.

Shared modules still used by the current demo:

- `clients.py`
- `config.py`
- `metrics.py`
- `pipeline.py`
- `prom_metrics.py`
- `reference.py`
- `reporting.py`

`drift.configs`

Shared configuration. It still contains some legacy runner settings and should be split later.

`drift.docs`

Current documentation and architecture notes. `architecture_notes` is intentionally not ignored by Git.

## Main Commands

```bash
make serve
make serve-synthetic
make generate-phase-demo-data
DRIFT_DEMO_OUTPUT_ROOT=drift/artifacts/runs make compute-demo-metrics
make monitoring-restart
```

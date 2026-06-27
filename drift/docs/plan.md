# Drift module current plan

## Current State

Актуальный рабочий контур для учебного demo:

```text
predict-api
synthetic-api
  -> drift.demo.phase_worker
  -> drift/artifacts/demo/events.sqlite
  -> drift.demo.metrics_worker
  -> drift/artifacts/runs/<run_id>_demo_debug/*.json*
  -> drift.monitoring exporter
  -> Prometheus
  -> Grafana
```

Старый batch `drift.runner.cli` больше не является основным demo flow. Он оставлен как legacy runner и как источник shared-модулей для текущего pipeline.

## Module Boundaries

`drift.demo`

Текущий offline demo orchestration:

- `phase_worker.py`: генерирует фазовый LLM-поток, вызывает `predict-api` и `synthetic-api`, пишет события в SQLite;
- `metrics_worker.py`: читает SQLite, считает окна и метрики, пишет JSON artifacts для exporter.

`drift.store`

SQLite event store. Это основной source of truth для demo events.

`drift.replay`

Replay fallback/cache. Используется, когда нужно повторить demo без LLM generation.

`drift.online`

Experimental заготовка online synthetic режима. Не считать production-ready частью текущего acceptance flow.

`drift.monitoring`

FastAPI exporter для Prometheus. Сейчас читает artifact files, а не SQLite напрямую.

`drift.synthetic_api`

Отдельный FastAPI сервис с ролями `generator` и `expert`.

`drift.runner`

Legacy/shared слой. Целиком удалять нельзя, потому что текущий demo использует:

- `drift.runner.clients`
- `drift.runner.config`
- `drift.runner.metrics`
- `drift.runner.pipeline`
- `drift.runner.prom_metrics`
- `drift.runner.reference`
- `drift.runner.reporting`

## Canonical Commands

```bash
conda activate ML_Ops
make serve
make serve-synthetic
make generate-phase-demo-data
DRIFT_DEMO_OUTPUT_ROOT=drift/artifacts/runs make compute-demo-metrics
make monitoring-restart
```

Replay fallback:

```bash
DRIFT_REPLAY_DATASET=drift/replay/messages.generated.jsonl make run-replay-demo
DRIFT_DEMO_OUTPUT_ROOT=drift/artifacts/runs make compute-demo-metrics
make monitoring-restart
```

## Current Phase Scenario

Canonical hidden phase names:

```text
A_baseline
B_lexical
C_label_distribution
D_association
A_recovery
```

Фаза хранится в SQLite/debug artifacts. В Prometheus/Grafana phase не является общим label и остается только в `drift_window_metric_status_code`.

## Cleanup Plan

1. Держать текущий demo код в `drift.demo`, а legacy batch runner в `drift.runner`.
2. Не добавлять новую функциональность в `drift.runner.cli`, кроме поддержки regression tests.
3. После стабилизации online режима решить судьбу `drift.online`: оставить как отдельный режим или перенести под `drift.demo.online`.
4. Разделить `drift/configs/runner.yaml` на shared config и legacy batch config.
5. Удалять старые artifacts из `drift/artifacts`, но не хранить runtime outputs в Git.
6. Держать актуальную документацию в `drift/docs`; `drift/docs/architecture_notes` теперь не должен быть в `.gitignore`.

## Acceptance Tests

Основной regression набор:

```bash
TMPDIR=/tmp python -m pytest -s tests/test_demo_monitoring_pipeline.py tests/test_drift_runner.py tests/test_drift_monitoring.py tests/test_synthetic_api.py tests/test_synthetic_prompts.py -q
```

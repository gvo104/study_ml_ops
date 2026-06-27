# Phase LLM demo monitoring pipeline

## Текущий flow

```text
drift.demo.phase_worker
  -> synthetic-api role=generator, phase=A/B/C/D, target_label=hidden
  -> predict-api /predict
  -> synthetic-api role=expert, без phase и target_label
  -> SQLite prediction_events
  -> drift.demo.metrics_worker
  -> prom_metrics_latest.json + window_metrics.jsonl
  -> drift-exporter -> Prometheus -> Grafana
```

`hidden_phase` и `hidden_target_label` сохраняются в SQLite/debug artifacts для offline-разбора. В Prometheus/Grafana фаза не является общим label и не участвует в фильтрах.

## Команды

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

Полезные env overrides:

```bash
DRIFT_EVENTS_DB=drift/artifacts/demo/events.sqlite
DRIFT_PHASE_INTERVAL_SECONDS=0
DRIFT_GENERATED_REPLAY_DATASET=drift/replay/messages.generated.jsonl
DRIFT_DEMO_WINDOW_SIZE=20
DRIFT_DEMO_STEP_SIZE=20
```

## Фазы

Default scenario:

```text
A_baseline: 40 events
B_lexical: 20 events
C_label_distribution: 20 events
D_association: 20 events
A_recovery: 20 events
```

Окна для demo выровнены по фазам: `window_size=20`, `step_size=20`.

- `A_baseline`: baseline-like wording, ожидается `ok`.
- `B_lexical`: меняется словарь/стиль, основной маркер `token_distribution_jsd`.
- `C_label_distribution`: baseline-like wording, но меняется частота target labels, основной маркер `target_distribution_jsd`.
- `D_association`: знакомые слова используются в другом контексте, основной маркер `token_label_association_drift`.
- `A_recovery`: возврат к baseline-like трафику.

## Baseline calibration

Metrics worker калибрует thresholds только по чистым baseline-окнам, где все events имеют `hidden_phase=A_baseline`, `normal`, `A` или `baseline`.

Если чистых baseline-окон нет, output получает:

```text
threshold_calibration_status=insufficient_baseline
overall_status=insufficient_data
```

Thresholds считаются как baseline envelope:

```text
higher_is_worse: warning=max_baseline+margin, critical=max_baseline+stronger_margin
lower_is_worse: warning=min_baseline-margin, critical=min_baseline-stronger_margin
```

Сравнение строгое: `>` для higher-is-worse и `<` для lower-is-worse. Поэтому baseline A не должен становиться `critical` только из-за равенства порогу.

## Prometheus/Grafana contract

Фаза экспортируется только здесь:

```promql
drift_window_metric_status_code{run_id=~"$run_id"}
```

Во всех остальных метриках labels содержат только monitoring labels без `phase`: `run_id`, `mode`, `status`, `window_index` при оконных метриках.

Grafana больше не содержит panel `Current Traffic Scenario`, phase variable и phase legends. Фазу можно смотреть только в `Window Metric Status Table`, чтобы проверить учебный сценарий, но не использовать как объясняющий сигнал мониторинга.

## Проверка

```bash
make generate-phase-demo-data
DRIFT_DEMO_OUTPUT_ROOT=drift/artifacts/runs make compute-demo-metrics
make monitoring-restart
```

Что смотреть:

- В SQLite есть события всех фаз, но dashboard не показывает phase отдельно.
- Первые A baseline окна имеют `ok` или близкий к `ok` статус.
- На B растет прежде всего `token_distribution_jsd`.
- На C растет прежде всего `target_distribution_jsd`.
- На D появляется сигнал по `token_label_association_drift`, если хватает labeled examples.
- В Prometheus `phase=` встречается только в строках `drift_window_metric_status_code`.

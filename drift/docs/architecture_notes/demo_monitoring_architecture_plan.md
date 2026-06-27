# Demo monitoring event-store architecture

## Purpose

Эта заметка фиксирует текущую архитектуру demo monitoring pipeline через event store.

Основная цель: имитировать production-like поток пользовательских сообщений, считать drift/quality по окнам и показывать результат в Prometheus/Grafana без раскрытия hidden phase как monitoring signal.

## Current Architecture

```text
predict-api
synthetic-api
  -> drift.demo.phase_worker
  -> SQLite prediction_events
  -> drift.demo.metrics_worker
  -> JSON artifacts
  -> drift.monitoring exporter
  -> Prometheus
  -> Grafana
```

`predict-api` не знает про фазы и не загружает LLM.

`synthetic-api` отвечает только за LLM роли `generator` и `expert`.

`SQLite prediction_events` - центральный event store для demo events.

`drift.demo.metrics_worker` - единственный текущий writer monitoring artifacts для offline demo.

## Event Store

Основная таблица:

```text
prediction_events
```

Ключевые группы полей:

- raw user message: `raw_text`;
- production preprocessing features: `tokens_stemmed`, `num_of_characters`, `num_of_sentences`;
- classifier output: `model_prediction`, `model_confidence`, `model_probabilities_json`;
- expert proxy truth: `expert_label`, `expert_confidence`, `expert_reason`, `expert_labeled_at`;
- hidden/debug metadata: `hidden_phase`, `hidden_target_label`, `metadata_json`;
- retraining flags: `is_selected_for_expert`, `is_training_candidate`.

`event_id` уникальный, поэтому replay/generation можно перезапускать без дублей.

## Offline Phase Demo

Основной worker:

```text
drift.demo.phase_worker
```

Default scenario:

```text
A_baseline: 40
B_lexical: 20
C_label_distribution: 20
D_association: 20
A_recovery: 20
```

Для каждого event:

1. worker выбирает hidden phase и hidden target label;
2. вызывает `synthetic-api` с `role=generator`;
3. отправляет текст в `predict-api`;
4. вызывает `synthetic-api` с `role=expert`;
5. прогоняет текст через production preprocessing;
6. сохраняет событие в SQLite;
7. пишет replay cache в `drift/replay/messages.generated.jsonl`.

Reject/regenerate не используется в runtime pipeline. Expert label считается proxy truth.

## Replay Fallback

`drift.replay.worker` читает JSONL dataset и пишет события в тот же SQLite store.

Основной generated cache:

```text
drift/replay/messages.generated.jsonl
```

Replay нужен для воспроизводимой демонстрации без повторной LLM generation.

## Online Synthetic Status

`drift.online.traffic_worker` и `drift.online.expert_worker` существуют как experimental online mode.

Они не являются основным acceptance flow. Перед включением в основной demo нужно отдельно проверить batch labeling, scheduling и failure handling.

## Metrics

Metrics worker:

```text
drift.demo.metrics_worker
```

Считает:

- `token_distribution_jsd`;
- `model_prediction_distribution_jsd`;
- `target_distribution_jsd`;
- `model_confidence_mean`;
- `expert_confidence_mean`;
- `model_expert_disagreement_rate`;
- `model_expert_macro_f1`;
- `token_label_association_drift`.

Quality metrics требуют `expert_label`.

Association drift получает `insufficient_data`, если в окне недостаточно labeled examples/classes.

## Windows And Calibration

Offline demo defaults:

```text
window_size=20
step_size=20
```

Thresholds калибруются только по чистым baseline windows:

```text
hidden_phase in {"A_baseline", "normal", "A", "baseline"}
```

Если baseline windows нет:

```text
threshold_calibration_status=insufficient_baseline
overall_status=insufficient_data
```

Threshold model:

```text
higher_is_worse: warning=max_baseline+margin, critical=max_baseline+stronger_margin
lower_is_worse: warning=min_baseline-margin, critical=min_baseline-stronger_margin
```

Сравнение строгое: `>` или `<`, чтобы baseline equality не давала ложный critical.

## Monitoring Contract

Exporter остается artifact-based:

```text
SQLite events -> metrics worker -> prom_metrics_latest.json/window_metrics.jsonl -> exporter
```

`phase` не является общим Prometheus label.

Разрешенное место, где phase видна для учебной проверки:

```promql
drift_window_metric_status_code{run_id=~"$run_id"}
```

Grafana не должна иметь:

- `Current Traffic Scenario` panel;
- phase filter;
- phase legends в общих графиках.

## Retraining Data

Если окно получает `critical`, metrics worker помечает labeled events из этого окна:

```text
is_training_candidate=true
```

Export:

```bash
make export-training-candidates
```

Hidden fields не экспортируются по умолчанию.

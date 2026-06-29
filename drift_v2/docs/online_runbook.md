# Drift V2 Online Runbook

## Purpose

`drift_v2 online` is the clean online demo path.

Flow:

```text
synthetic-api generator
  -> predict-api /predict
  -> SQLite prediction_events (clean run-scoped source_mode)
  -> synthetic-api expert
  -> SQLite expert updates
  -> drift_v2 window metrics
  -> stable clean artifacts
  -> drift_v2 exporter -> Prometheus -> Grafana
```

## Main Commands

Start one baseline batch:

```bash
make drift-v2-online-seed-baseline
```

Run one tick for the current phase:

```bash
DRIFT_V2_HIDDEN_PHASE=B_lexical make drift-v2-online-tick
```

Start infinite online loop:

```bash
make restart-drift-v2-online-monitoring
```

Inspect clean online progress:

```bash
make show-drift-v2-online-progress
```

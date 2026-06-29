# Drift V2 Groundwork

`drift_v2` is a clean rebuild track placed next to the legacy `drift` module.

Current scope:

- read immutable demo events from the existing SQLite store;
- define explicit event-level and window-level contracts;
- make window construction deterministic for offline playback;
- run a clean online synthetic loop next to offline playback.

## Data Boundaries

Event-level data comes from SQLite `prediction_events`.

Each `DemoEvent` carries only the fields we currently need to reason about the new pipeline:

- event identity and ordering: `event_index`, `event_id`, `created_at`
- request payload: `raw_text`, `tokens_stemmed`, `num_of_characters`, `num_of_sentences`
- model outputs: `model_prediction`, `model_confidence`
- expert outputs: `expert_label`, `expert_confidence`
- hidden demo labels: `hidden_phase`, `hidden_target_label`

This gives us a strict split:

- per-event layer:
  - model prediction
  - model confidence
  - expert label
  - expert confidence
- per-window layer:
  - all drift and quality aggregates
  - window status
  - baseline calibration

## Window Rules

Offline playback uses a visible prefix of the event stream.

For `window_size=20` and `step_size=20`:

- `visible_events < 20` -> `0` complete windows
- `visible_events = 20` -> `1` complete window
- `visible_events = 40` -> `2` complete windows
- `visible_events = 120` -> `6` complete windows

This matches the intended demo semantics: a window is computed only after its full batch of events has arrived.

## Online Rules

Online mode appends new synthetic events into SQLite under a clean run-scoped `source_mode`.

- offline:
  - immutable SQLite stream
  - visible prefix replay
- online:
  - append-only SQLite stream
  - visible events equal current run event count

Online loop order per tick:

1. generator creates synthetic requests
2. `predict-api` writes model outputs into SQLite
3. expert worker labels recent unlabeled events of the same clean `source_mode`
4. v2 window metrics are recomputed from full windows only
5. one stable run directory is fully rewritten

## Next Layer

Once this groundwork is accepted, the next step is to add:

1. window metric computation on top of `WindowBatch`
2. stable artifact writer for a single demo run
3. Prometheus exporter for:
   - recent event-level outputs
   - window-level aggregates
4. Grafana panels built on those two separate layers

# Offline demo monitoring system inventory

## Scope

Этот документ фиксирует текущее состояние offline/demo monitoring pipeline после перехода на SQLite event store и phase LLM generator.

Основной рабочий режим сейчас: `offline phase demo`.

```text
predict-api + synthetic-api
  -> drift.demo.phase_worker
  -> SQLite prediction_events
  -> drift.demo.metrics_worker
  -> JSON artifacts
  -> drift-exporter
  -> Prometheus
  -> Grafana
```

Online synthetic режим добавлен как заготовка, но не является основным проверенным demo flow.

## Рабочий offline flow

1. `make serve`

   Поднимает основной классификатор на `http://127.0.0.1:8000`. Он принимает текст и возвращает `prediction`, `confidence`, `probabilities`.

2. `make serve-synthetic`

   Поднимает `synthetic-api` на `http://127.0.0.1:8001`.

   API используется в двух ролях:

   - `generator`: генерирует пользовательский текст по фазе `A/B/C/D` и hidden target label;
   - `expert`: размечает текст без знания phase и target label.

3. `make generate-phase-demo-data`

   Запускает `drift.demo.phase_worker`.

   Worker проходит сценарий:

   ```text
   A_baseline: 40
   B_lexical: 20
   C_label_distribution: 20
   D_association: 20
   A_recovery: 20
   ```

   Для каждого события:

   - выбирает hidden phase;
   - выбирает hidden target label;
   - вызывает synthetic generator;
   - отправляет текст в predict-api;
   - вызывает synthetic expert;
   - сохраняет событие в SQLite;
   - пишет replay cache в `drift/replay/messages.generated.jsonl`.

   Основная БД:

   ```text
   drift/artifacts/demo/events.sqlite
   ```

4. `DRIFT_DEMO_OUTPUT_ROOT=drift/artifacts/runs make compute-demo-metrics`

   Запускает `drift.demo.metrics_worker`.

   Worker:

   - читает события из SQLite;
   - режет поток на окна `window_size=20`, `step_size=20`;
   - калибрует thresholds только по чистым baseline-окнам;
   - считает drift и quality metrics;
   - помечает labeled events из critical окон как `is_training_candidate`;
   - пишет artifacts для exporter.

   Основные artifacts:

   ```text
   drift/artifacts/runs/<run_id>_demo_debug/window_metrics.jsonl
   drift/artifacts/runs/<run_id>_demo_debug/run_summary.json
   drift/artifacts/runs/<run_id>_demo_debug/prom_metrics_latest.json
   ```

5. `make monitoring-restart`

   Перезапускает monitoring stack:

   ```text
   drift-exporter -> Prometheus -> Grafana
   ```

   Exporter читает JSON artifacts из `drift/artifacts/runs`.

## Что сейчас точно используется и нужно

`src.api.app`

Основной inference API. Нужен для production-like поведения: все demo events проходят через тот же API классификатора, а не через прямой вызов модели.

`drift.synthetic_api`

Нужен для генератора и эксперта. Сейчас это центральный источник synthetic traffic и proxy labels.

Ключевые prompt files:

```text
drift/synthetic_api/prompt_templates/phase_definitions.txt
drift/synthetic_api/prompt_templates/generator_user.txt
drift/synthetic_api/prompt_templates/expert_user.txt
drift/synthetic_api/prompt_templates/label_boundaries.txt
drift/synthetic_api/prompt_templates/status_definitions.txt
```

`drift.demo.phase_worker`

Основной orchestrator offline demo. Он создает фазовый поток и наполняет SQLite.

`drift.store`

SQLite event store. Сейчас это центральный источник truth для demo monitoring:

```text
drift/store/schema.sql
drift/store/db.py
```

`drift.demo.metrics_worker`

Основной расчетчик demo metrics из SQLite в JSON artifacts.

Использует общие функции из `drift.runner.metrics`, `drift.runner.reference`, `drift.runner.prom_metrics`.

`drift.monitoring`

Exporter FastAPI. Нужен для Prometheus. Важный текущий контракт: `phase` не экспортируется как общий label и остается только в `drift_window_metric_status_code`.

`monitoring/grafana`

Текущий dashboard для demo monitoring. Панель `Current Traffic Scenario` удалена; phase filter не должен возвращаться.

`docker-compose.monitoring.yml`

Нужен для локального monitoring stack. Сейчас exporter монтирует `./drift/artifacts/runs` read-only.

`drift/replay`

Нужен как fallback/offline cache. Основной replay file после генерации:

```text
drift/replay/messages.generated.jsonl
```

Replay полезен, когда нужно повторить demo без LLM generation.

## Что готово, но требует аккуратной проверки

`drift.online.traffic_worker` и `drift.online.expert_worker`

Это заготовка online synthetic режима. Логика есть:

```text
traffic worker -> SQLite unlabeled events
expert batch -> labels recent unlabeled events
metrics worker -> artifacts
```

Но это еще не основной acceptance flow. Перед тем как считать готовым, нужно проверить:

- batch expert selection на реальном тайминге;
- поведение при недоступном synthetic-api;
- отсутствие phase leakage в online artifacts;
- удобные make-команды для циклического запуска;
- отдельные integration tests на online сценарий.

`drift.runner.cli`

Старый batch drift-runner пока оставлен. Он не основной для нового demo, но его нельзя сразу удалять, потому что от `drift.runner` еще используются общие модули:

```text
drift.runner.metrics
drift.runner.pipeline
drift.runner.reference
drift.runner.prom_metrics
drift.runner.reporting
drift.runner.config
drift.runner.clients
```

Нужно позже разделить:

- оставить shared modules;
- старый scenario/batch CLI либо архивировать, либо явно назвать legacy.

`drift/configs/runner.yaml`

Используется новым pipeline для API URLs, allowed labels, reference settings и thresholds-related params.

Но в нем еще есть legacy поля старого runner:

```text
max_generation_attempts_per_sample
target_shift_distribution
profiles.*.target_label_strategy
profiles.*.phase_counts с C_target_shift
```

Их нужно либо удалить после миграции, либо отделить в legacy config.

## Что нужно почистить

Artifacts:

```text
drift/artifacts/runs_archive/*
drift/artifacts/demo/events.sqlite
drift/artifacts/runs/<old_run_id>/*
```

Это runtime outputs. Их не стоит держать как часть чистой архитектуры. Для локального demo можно оставлять, но перед фиксацией проекта лучше:

- убедиться, что artifacts игнорируются git;
- оставить только маленький synthetic replay пример;
- не хранить старые generated metrics как source of truth.

Docs:

```text
drift/docs/architecture_notes/*
```

Эта папка содержит актуальные architecture notes и больше не должна быть в `.gitignore`.

Legacy docs:

```text
drift/docs/plan.md
drift/docs/llm_sint_and_exp.md
```

Нужно перечитать и разделить:

- актуальные инструкции оставить;
- устаревшие планы пометить как legacy или удалить;
- не держать несколько противоречащих описаний flow.

Makefile:

Сейчас есть команды и для старого runner, и для нового demo:

```text
run-drift-debug
run-drift-full
run-replay-demo
generate-phase-demo-data
compute-demo-metrics
run-online-traffic
run-expert-batch
```

Нужно явно разделить секции:

- current offline demo;
- replay fallback;
- online experimental;
- legacy runner.

## Что уже не является основным и может быть кандидатом на удаление

`drift.runner.scenario`

Используется старым batch runner. Новый offline demo выбирает фазы в `drift.demo.phase_worker`.

Удалять можно только после решения по старому `drift.runner.cli`.

Old batch artifacts:

```text
accepted_samples.jsonl
rejected_samples.jsonl
label_diagnostics.json
run_report.md
```

Это outputs старого reject/regenerate pipeline. В новом monitoring pipeline reject/regenerate не используется, expert label считается proxy truth.

Старые фазы и названия:

```text
C_target_shift
target_shift_distribution
label_distribution_shift
association_drift
lexical_drift
normal
```

В новом phase demo canonical names:

```text
A_baseline
B_lexical
C_label_distribution
D_association
A_recovery
```

Старые названия могут оставаться только как backward compatibility в БД/debug коде. В документации и новых tests лучше использовать canonical names.

## Что нужно доработать ближайшим шагом

1. Добавить `make` команды для просмотра SQLite demo state:

   ```text
   show-demo-events
   show-demo-progress
   watch-demo-progress
   ```

2. Вынести demo scenario counts в config или CLI options для `phase_worker`, чтобы не держать `40/20/20/20/20` только в коде.

3. Добавить итоговую docs-страницу `drift/docs/offline_demo_runbook.md` с одним стабильным запуском и проверками.

4. Привести `drift/configs/runner.yaml` к двум секциям: shared settings и legacy runner settings.

5. Решить судьбу `drift.runner.cli`: оставить как `legacy-drift-runner` или удалить после полной миграции tests.

6. Проверить real LLM generation quality по фазам, особенно:

   - B должен менять лексику без смены label distribution;
   - C должен менять только label frequency;
   - D должен давать context-based association shift.

7. После стабилизации удалить или архивировать старые artifacts из `drift/artifacts/runs_archive`.

## Текущее acceptance состояние

Проверенный test set:

```bash
conda activate ML_Ops
TMPDIR=/tmp python -m pytest -s tests/test_demo_monitoring_pipeline.py tests/test_drift_runner.py tests/test_drift_monitoring.py tests/test_synthetic_api.py tests/test_synthetic_prompts.py -q
```

Последний результат:

```text
46 passed
```

Отдельно после добавления progress output в phase worker:

```text
21 passed
```

## Короткий вывод

Полностью готовый и нужный сейчас контур: offline phase demo через `phase_worker -> SQLite -> metrics_worker -> artifacts -> exporter -> Prometheus/Grafana`.

Replay нужен как fallback/cache.

Online workers есть, но пока experimental.

Старый `drift.runner` не является основным flow, но пока содержит shared code, поэтому удалять его целиком нельзя. Чистку нужно начинать с artifacts, docs и naming/config debt, а не с удаления Python modules.

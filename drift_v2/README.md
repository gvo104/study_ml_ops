# drift_v2

`drift_v2` — отдельный демонстрационный контур мониторинга качества и дрейфа для проекта `study_ml_ops`. Он не заменяет основной FastAPI inference API и не встроен в него напрямую, а работает рядом с ним как monitoring-пайплайн с offline/online сценариями.

## Назначение

Подсистема нужна для следующих задач:

- считать оконные drift-метрики по потоку событий;
- воспроизводить накопленный event stream из SQLite;
- запускать synthetic online сценарий с generator/expert ролями;
- публиковать monitoring-метрики в Prometheus;
- отображать их в Grafana;
- сохранять run-артефакты и markdown-отчеты.

## Состав подсистемы

### Основные модули

- `cli.py`:
  вход в offline/online сценарии `drift_v2`.
- `pipeline.py`, `online.py`, `playback.py`, `windows.py`, `repository.py`:
  orchestration событий, окон и сценариев воспроизведения.
- `metrics.py`, `contracts.py`, `config.py`:
  контракты данных, расчет метрик и конфигурация.

### `runner/`

Пакет `runner` отвечает за production-like обработку monitoring run-а:

- читает reference statistics;
- строит окна событий;
- считает drift-метрики;
- пишет JSON-артефакты;
- формирует markdown report;
- собирает label diagnostics.

Ключевые артефакты run-а сохраняются в `drift_v2/artifacts/runs/`.

### `monitoring/`

Отдельный FastAPI exporter:

- `GET /health`
- `GET /metrics`

Exporter читает последние run snapshots и отдает их в формате Prometheus metrics.

### `synthetic_api/`

Это отдельный сервис для demo online сценария. Он нужен для генерации synthetic traffic и экспертных ответов, а не для основного пользовательского inference API проекта.

Внутри каталога находятся:

- FastAPI app;
- prompt templates;
- runtime adapters;
- frontend `index.html` для demo-сценария.

### `store/`

SQLite-backed event storage:

- `schema.sql` — схема базы;
- `db.py` — доступ к БД;
- `events.sqlite` — demo store с событиями.

### `reference/`

Reference statistics и reference dataset для сравнения окон:

- `metadata.json`
- `reference_stats.json`
- `reference_train.csv`

## Drift-метрики

В текущей реализации используются следующие ключевые метрики:

- `token_distribution_jsd`
- `model_prediction_distribution_jsd`
- `target_distribution_jsd`
- `model_expert_disagreement_rate`
- `model_expert_macro_f1`
- `token_label_association_drift`

Что это означает на практике:

- data drift покрывается через распределения токенов;
- target drift покрывается через распределение экспертных меток;
- proxy для concept drift покрывается через disagreement, macro F1 и изменение связи токенов с метками.

Это monitoring-реализация demo-уровня. Отдельного полноформатного production-модуля с явной таксономией concept drift в репозитории сейчас нет.

## Запуск

Все команды ниже предполагают:

```bash
conda activate ML_Ops
```

### Offline monitoring

```bash
make drift-monitoring-offline
```

Команда:

- поднимает Grafana, Prometheus и exporter;
- запускает offline replay по SQLite event store;
- обновляет run-артефакты в `drift_v2/artifacts/runs/`.

### Online synthetic monitoring

```bash
make drift-monitoring-online
```

Команда:

- поднимает monitoring stack;
- запускает synthetic online loop;
- пишет новые события и обновляет monitoring metrics.

### Synthetic API отдельно

```bash
make serve-synthetic
```

По умолчанию сервис поднимается на `http://localhost:8001`.

### Monitoring stack

Адреса после запуска:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- exporter metrics: `http://localhost:9208/metrics`
- exporter health: `http://localhost:9208/health`

Остановка:

```bash
make monitoring-down
```

## Артефакты и отчеты

Подсистема генерирует:

- JSON snapshots для monitoring;
- данные по оконным метрикам;
- diagnostics по меткам;
- markdown report `run_report.md`.

Markdown report содержит:

- идентификатор run-а;
- режим запуска;
- число окон;
- наиболее выраженные drift-сегменты;
- проблемные метки;
- показатели с недостатком данных.

## Ограничения

- `drift_v2` не интегрирован как единая runtime-часть основного API `src.api`.
- Offline и online сценарии ориентированы на demo и мониторинговую диагностику.
- Synthetic API не является пользовательским production endpoint.
- Monitoring зависит от структуры run-артефактов и не заменяет централизованную observability-платформу.

## Связанные файлы

- `docker-compose.monitoring.yml`
- `monitoring/prometheus/prometheus.yml`
- `monitoring/prometheus/drift_v2_rules.yml`
- `monitoring/grafana/dashboards/`
- `drift_v2/docs/online_runbook.md`

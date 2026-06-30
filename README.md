# study_ml_ops

MLOps-проект для классификации текстов по темам психического здоровья. Репозиторий объединяет основной ML-пайплайн обучения и инференса, FastAPI-сервис с web UI, трекинг экспериментов в MLflow, версионирование данных через DVC и отдельный демонстрационный контур drift-мониторинга `drift_v2`.

## Назначение проекта и текущий статус

Проект состоит из четырех основных подсистем:

1. Основной ML-контур:
   подготовка данных, обучение модели, сохранение артефактов, CLI-инференс и FastAPI API.
2. Экспериментальный контур:
   запуск нескольких конфигов моделей из `configs/experiments/` с логированием в MLflow.
3. Web UI:
   страницы инференса и экспериментов с последними предсказаниями, флагами аномалий, retrain-кнопкой и отображением MLflow-ранов.
4. Drift-monitoring контур `drift_v2`:
   офлайн/онлайн сценарии мониторинга, Prometheus exporter, Grafana dashboards, synthetic API и markdown-отчеты по окнам дрейфа.

Текущее состояние проекта:

- Основной train/inference/API/UI-контур реализован и покрыт тестами.
- MLflow и DVC интегрированы и используются в локальной инфраструктуре через Docker Compose.
- `drift_v2` реализован как отдельный демонстрационный monitoring-пайплайн, а не как полностью встроенный production-контур.
- Kubernetes и Argo CD присутствуют как частично рабочие инфраструктурные заготовки в основном для MinIO и MLflow.
- Документация в этом `README` и в `docs/` актуализирована по текущему состоянию кода и не предполагает несуществующих компонентов.

## Архитектура репозитория

```text
study_ml_ops/
├── src/                    # Основной production-like Python-код
├── drift_v2/               # Отдельный демонстрационный контур drift-monitoring
├── configs/                # Конфиги production и experiment runs
├── data/                   # Данные и DVC-описания
├── models/                 # Локальные артефакты обученной production-модели
├── monitoring/             # Prometheus/Grafana конфиги и dashboards
├── k8s/                    # Kubernetes manifests для инфраструктуры
├── arcd/                   # Argo CD Application manifest
├── docs/                   # Sphinx-документация
├── tests/                  # Тесты основного контура и drift_v2
├── .github/workflows/      # GitHub Actions пайплайны CI/CD
├── docker-compose.yml      # Локальная инфраструктура MLflow + MinIO + webapp
├── docker-compose.monitoring.yml
├── dvc.yaml                # DVC pipeline prepare -> train
└── Makefile                # Основные команды проекта
```

## Подробное описание модулей

### Основной пакет `src`

#### `src.api`

- `app.py`:
  создает FastAPI-приложение, UI-страницы `/` и `/experiments`, API endpoints `/predict`, `/health`, `/api/dashboard`, `/api/experiments`, `/api/retrain`.
- `bootstrap.py`:
  проверяет наличие датасета и артефактов модели, пытается подтянуть данные через DVC, при необходимости инициирует bootstrap-обучение.
- `dashboard.py`:
  хранит in-memory состояние UI, последние предсказания, простые drift-уведомления, статус retraining и агрегаты для страницы экспериментов.
- `schemas.py`:
  Pydantic-схемы запросов и ответов API.
- `ui.py`:
  HTML/CSS/JS для встроенного web UI без отдельного frontend-фреймворка.

#### `src.data`

- `make_dataset.py`:
  подготавливает `data/processed/fin_data.csv` из `data/raw/fin_data.csv` или другого переданного CSV.
- `loaders.py`:
  загружает и валидирует processed dataset.
- `validation.py`:
  проверяет обязательные колонки и корректность структуры набора данных.

#### `src.features`

- `preprocess.py`:
  функции предобработки текста.
- `numerical.py`:
  извлечение числовых признаков из текста.
- `build_features.py`:
  сборка признакового пайплайна: TF-IDF, SVD и числовые признаки для train/inference.

#### `src.models`

- `trainer.py`:
  основной train-loop: train/test split, oversampling, fit модели, оценка, графики, сохранение артефактов, логирование в MLflow.
- `train_model.py`:
  CLI entrypoint для одиночного запуска обучения.
- `run_experiments.py`:
  пакетный запуск всех YAML-конфигов из каталога экспериментов.
- `predict_model.py`:
  CLI-инференс по сохраненным артефактам.
- `tracking.py`:
  интеграция с MLflow, чтение экспериментов и логирование метрик/моделей/артефактов.
- `registry.py`:
  загрузка предиктора и локальных артефактов модели.
- `factory.py`, `xgboost_model.py`, `lightgbm_model.py`, `sklearn_models.py`, `base.py`:
  реестр поддерживаемых моделей и их адаптеры.
- `artifacts.py`:
  вспомогательные функции работы с артефактами модели.

#### `src.visualization`

- `visualize.py`:
  confusion matrix и текстовый classification report для train/eval артефактов.

#### Системные модули

- `src/config.py`:
  константы путей, имена колонок, настройки MLflow/S3.
- `src/config_loader.py`:
  загрузка YAML-конфигов обучения.
- `src/cli.py`:
  группа команд `study-mlops`.
- `src/utils.py`:
  служебные утилиты и логирование.

### Пакет `drift_v2`

#### `drift_v2.runner`

- вычисляет оконные drift-метрики;
- строит output-артефакты run-а;
- пишет markdown report и JSON-данные для мониторинга;
- поддерживает offline и online режимы исполнения.

#### `drift_v2.monitoring`

- FastAPI exporter с endpoint `/metrics`;
- читает последние артефакты `drift_v2` run-ов;
- публикует метрики для Prometheus и Grafana.

#### `drift_v2.synthetic_api`

- отдельный FastAPI-сервис для synthetic traffic;
- содержит prompt templates и рантаймы для generator/expert сценариев;
- нужен для demo online monitoring-потока, а не для основного inference API.

#### `drift_v2.store`

- SQLite-backed хранилище событий;
- схема таблиц и функции доступа к prediction/expert событиям.

#### Прочие модули `drift_v2`

- `cli.py`, `pipeline.py`, `online.py`, `playback.py`, `repository.py`, `windows.py`, `metrics.py`, `config.py`, `contracts.py`:
  orchestration offline/online контуров, окна событий, reference statistics и контракты данных.

## Данные и модель

### Датасет

- Основной dataset:
  `data/processed/fin_data.csv`
- DVC-описание processed dataset:
  `data/processed/fin_data.csv.dvc`
- DVC pipeline:
  `dvc.yaml`

Ожидаемые ключевые колонки:

- `tokens_stemmed`
- `status`
- `num_of_characters`
- `num_of_sentences`

Исходные данные могут быть:

- подготовлены локально через `make data`;
- скачаны из DVC remote через `dvc pull`;
- использованы из `data/raw/fin_data.csv`.

### Базовая модель

Production baseline задается в `configs/xgboost_baseline.yaml`:

- модель: `xgboost`
- признаки: TF-IDF по `tokens_stemmed` + SVD + числовые признаки
- train-опции: `RandomOverSampler`, `test_size=0.2`, `random_state=101`

### Поддерживаемые модели

Через YAML-конфиги поддерживаются:

- `xgboost`
- `lightgbm`
- `logistic_regression`
- `random_forest`

Каталог experiment-конфигов:

- `configs/experiments/`

## Возможности проекта

### Основной контур

- подготовка processed dataset;
- обучение baseline-модели и сохранение артефактов в `models/xgboost/`;
- логирование экспериментов и артефактов в MLflow;
- CLI и HTTP-инференс;
- встроенный web UI для инференса и экспериментов;
- in-memory мониторинг последних предсказаний;
- простые флаги аномалий:
  `low_confidence`, `too_short`, `too_long`;
- кнопка retraining из UI;
- чтение списка MLflow runs в UI.

### Контур `drift_v2`

- офлайн воспроизведение окон событий из SQLite;
- онлайн synthetic traffic loop;
- расчет:
  `token_distribution_jsd`,
  `model_prediction_distribution_jsd`,
  `target_distribution_jsd`,
  `model_expert_disagreement_rate`,
  `model_expert_macro_f1`,
  `token_label_association_drift`;
- markdown-отчеты по run-ам;
- Prometheus exporter;
- Grafana dashboards;
- synthetic API для демо-сценариев.

### Что считать production-like, а что demo

Production-like:

- основной train/inference/API/UI-контур;
- MLflow/DVC локальная инфраструктура;
- Docker-образ API.

Demo / experimental:

- `drift_v2` synthetic online loop;
- drift-мониторинг как отдельный контур;
- Kubernetes/Argo CD интеграция;
- CI/CD deployment стадия.

## Инструкция по запуску

### 1. Подготовка окружения

```bash
conda env update --name ML_Ops --file environment.yml --prune
conda activate ML_Ops
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Проверка интерпретатора:

```bash
python --version
```

### 2. Локальный запуск без контейнеров

Подготовить датасет:

```bash
conda activate ML_Ops
make data
```

Обучить baseline-модель:

```bash
conda activate ML_Ops
make train
```

Сделать CLI-инференс:

```bash
conda activate ML_Ops
make predict TEXT="I feel sad and anxious and cannot sleep."
```

### 3. Инфраструктура MLflow и MinIO

Запуск:

```bash
conda activate ML_Ops
cp .env.example .env
make infra-up
```

Полезные адреса:

- FastAPI + UI: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`
- MinIO Console: `http://localhost:9001`
- MinIO S3 API: `http://localhost:9000`

Остановка:

```bash
conda activate ML_Ops
make infra-down
```

### 4. Запуск FastAPI и web UI

Локальный запуск API:

```bash
conda activate ML_Ops
make serve
```

После запуска доступны:

- `GET /health`
- `POST /predict`
- `GET /`
- `GET /experiments`
- `GET /docs`
- `GET /openapi.json`

### 5. Запуск тестов и проверок

Тесты:

```bash
conda activate ML_Ops
make test
```

Только synthetic API тесты:

```bash
conda activate ML_Ops
make test-synthetic
```

Линтер:

```bash
conda activate ML_Ops
make lint
```

### 6. Эксперименты

Запуск всех experiment-конфигов:

```bash
conda activate ML_Ops
make experiments
```

CLI-эквивалент:

```bash
conda activate ML_Ops
python -m src.models.run_experiments --data-path data/processed/fin_data.csv --configs-dir configs/experiments
```

### 7. DVC

Воспроизвести pipeline:

```bash
conda activate ML_Ops
make dvc-repro
```

Скачать данные из remote:

```bash
conda activate ML_Ops
make dvc-pull
```

### 8. Drift-monitoring

Запуск monitoring stack:

```bash
conda activate ML_Ops
make drift-monitoring-offline
```

Или online synthetic режим:

```bash
conda activate ML_Ops
make drift-monitoring-online
```

Адреса monitoring stack:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Drift exporter: `http://localhost:9208/metrics`

Остановка:

```bash
conda activate ML_Ops
make monitoring-down
```

### 9. Synthetic API

Запуск отдельного synthetic API:

```bash
conda activate ML_Ops
make serve-synthetic
```

По умолчанию сервис поднимается на `http://localhost:8001`.

## API и интерфейсы

### Основной HTTP API

Основные endpoints:

- `GET /health`:
  статус загрузки модели и базовых метаданных.
- `POST /predict`:
  инференс по JSON `{"text": "..."}`.
- `GET /api/dashboard`:
  последние предсказания и drift notifications для UI.
- `GET /api/experiments`:
  состояние модели, MLflow runs, retraining status, experiment configs.
- `POST /api/retrain`:
  фоновый retrain текущей production-конфигурации.

### Страницы UI

- `/`:
  инференс, последние предсказания, anomaly flags, drift notifications.
- `/experiments`:
  сведения о production-модели, retraining status, MLflow runs, experiment configs.

### CLI `study-mlops`

Доступные команды:

- `study-mlops data`
- `study-mlops train`
- `study-mlops experiments`
- `study-mlops predict`
- `study-mlops infra status`

### DVC интерфейс

- `dvc.yaml` содержит стадии `prepare` и `train`;
- remote настроен в `.dvc/config` на S3-совместимое хранилище MinIO.

### Monitoring интерфейс

- Prometheus читает `drift-v2-exporter:9208`;
- Grafana dashboards загружаются из `monitoring/grafana/dashboards/`;
- exporter читает JSON-артефакты последних `drift_v2` run-ов.

## Инфраструктура и деплой

### Что реально готово через Docker Compose

`docker-compose.yml` поднимает:

- MinIO;
- init-контейнер для bucket `ml-team`;
- MLflow server;
- webapp с FastAPI/UI.

`docker-compose.monitoring.yml` поднимает:

- drift exporter;
- Prometheus;
- Grafana.

### Что есть в Kubernetes

В `k8s/base/` есть manifests для:

- namespace;
- ConfigMap и Secret;
- MinIO deployment/service;
- init job `minio-init`;
- MLflow deployment/service.

Это инфраструктура для MinIO и MLflow. Kubernetes-манифестов для самого FastAPI/UI сервиса в репозитории сейчас нет.

### Что есть в Argo CD

В `arcd/ml-team-application.yml` есть Argo CD Application, указывающий на `k8s/base`. Это заготовка GitOps-деплоя инфраструктуры, а не полный CD всего приложения.

### Что есть в CI/CD

В `.github/workflows/` присутствуют:

- `ci-cd.yml`:
  lint, tests, build Python package, build Docker image, smoke deployment в PR на `main`.
- `ci.yml`:
  отдельный pipeline с tests, lint, Docker build для MLflow image и проверкой импорта MLflow.
- `deploy.yml`:
  попытка интеграции с Argo CD/Kubernetes.

Важно:

- CI настроен частично и местами дублируется.
- CD нельзя считать полностью рабочим production-потоком.
- `deploy.yml` выглядит как заготовка и требует доработки секретов и реальной среды.


## Связанная документация

- [Sphinx docs index](docs/index.rst)
- [Getting started](docs/getting-started.rst)
- [Commands](docs/commands.rst)
- [Project structure](docs/project-structure.rst)
- [drift_v2 README](drift_v2/README.md)
- [k8s README](k8s/README.md)

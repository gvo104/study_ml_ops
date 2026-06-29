# Primary_Medical_Consultation

MLOps-проект для классификации текстов по теме психического здоровья.

Production-код — в `src/`. Документация: [`docs/`](docs/) (Sphinx + MD-гайды).

## Быстрый старт

```bash
conda env update --name ML_Ops --file environment.yml --prune
conda activate ML_Ops
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install lightgbm    # для LightGBM-экспериментов, если нет в conda

cp .env.example .env    # опционально, MLflow/MinIO
make infra-up           # опционально
make data
make train
make predict TEXT="I feel sad and anxious and cannot sleep."
```

Датасет: `data/processed/fin_data.csv` (или `dvc pull`).

## Команды

| Команда | Описание |
|---------|----------|
| `make data` | Подготовка `data/processed/fin_data.csv` |
| `make train` | Production-обучение → `models/xgboost/` + MLflow |
| `make experiments` | 9 экспериментов из `configs/experiments/` → MLflow |
| `make predict TEXT="..."` | CLI-инференс |
| `make test` | pytest |
| `make serve` | FastAPI http://localhost:8000 |
| `make infra-up` | MinIO + MLflow + web UI (Docker) |
| `make dvc-repro` | DVC: prepare → train |
| `study-mlops experiments` | То же, что `make experiments` |

## Модели

В YAML-конфигах поле `model.name`:

| name | Описание |
|------|----------|
| `xgboost` | Основная production-модель |
| `lightgbm` | Gradient boosting (нужен `lightgbm`) |
| `logistic_regression` | Линейный baseline |
| `random_forest` | Ансамбль деревьев |

Конфиги экспериментов: [`configs/experiments/`](configs/experiments/).

## Структура

```text
study_ml_ops/
├── configs/                 # xgboost_baseline.yaml + experiments/
├── data/
├── models/xgboost/          # артефакты после make train
├── reports/figures/         # confusion matrix (production)
├── src/
│   ├── config.py, config_loader.py, cli.py
│   ├── data/, features/, models/, api/, visualization/
├── tests/
├── dvc.yaml
└── docs/
```

## Пайплайн признаков

- Текст: `tokens_stemmed` (без повторной предобработки по умолчанию)
- Числовые: `num_of_characters`, `num_of_sentences`
- TF-IDF (1,2) → TruncatedSVD → RandomOverSampler → классификатор
- Инференс: `FeatureBuilder.transform_single()` + `feature_builder.pkl`

## MLOps

- **MLflow** — метрики, модель, `evaluation/confusion_matrix.png` ([гайд](docs/mlflow-minio-setup.md))
- **DVC** — версионирование данных (`dvc pull` / `dvc push`)
- **API** — `POST /predict` с `{"text": "..."}`
- **Web UI** — `http://localhost:8000/` и `http://localhost:8000/experiments` через `docker compose up --build`

## Документация

| Файл | Содержание |
|------|------------|
| [getting-started.rst](docs/getting-started.rst) | Установка, train, experiments, API |
| [commands.rst](docs/commands.rst) | Makefile и CLI |
| [project-structure.rst](docs/project-structure.rst) | Модули `src/` |
| [mlflow-minio-setup.md](docs/mlflow-minio-setup.md) | Инфраструктура и эксперименты |
| [refactoring-plan.md](docs/refactoring-plan.md) | План рефакторинга |

---

<p><small>Based on the <a href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science</a> template.</small></p>

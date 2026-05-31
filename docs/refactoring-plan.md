# План рефакторинга study_ml_ops

> Статус: **завершён** (фаза 5 — переименование пакета `src` → `mh_ops` — отложена)

## Цель

Production-ready MLOps-контур: данные → признаки → обучение → инференс, MinIO / MLflow / DVC, модульный код, тесты, API, пакетные эксперименты.

## Выполнено

| Фаза | Содержание |
|------|------------|
| 0 | `.env.example`, dotenv в `config.py`, `docs/refactoring-plan.md` |
| 1 | `numerical.py`, loaders/validation, `FeatureBuilder`, `feature_builder.pkl` |
| 2 | `tracking.py`, `dvc.yaml`, Makefile infra/dvc |
| 3 | `trainer`, `registry`, `factory`, YAML-конфиги, `study-mlops` CLI |
| 4 | FastAPI, pytest, `run_experiments`, 9 experiment YAML, docs |

## Структура `src/`

```text
src/
├── config.py, config_loader.py, cli.py
├── data/          make_dataset, loaders, validation
├── features/      preprocess, numerical, build_features
├── models/        trainer, registry, tracking, factory, run_experiments
├── api/           FastAPI
└── visualization/ confusion matrix, reports
```

## Команды

```bash
make train              # production → models/xgboost/
make experiments        # configs/experiments/* → MLflow
make predict TEXT="..."
make test && make serve
make infra-up && make dvc-repro
```

## MLflow-артефакты на run

- Метрики: `accuracy`, `macro_f1`, `weighted_f1`
- Модель: `model/` (flavor зависит от алгоритма)
- `evaluation/confusion_matrix.png`
- `evaluation/classification_report.txt`
- `model_bundle/` — локальные pkl (если `save_local_artifacts: true`)

## Definition of Done

- [x] `make train && make predict` с `feature_builder.pkl`
- [x] `make experiments` + артефакты в MLflow
- [x] `make test`, `make serve`
- [x] Документация (README, Sphinx, MD-гайды)
- [ ] `dvc repro` на машине с настроенным remote
- [ ] Переименование пакета `src` (фаза 5)

## Отложено

- Пакет `mh_ops` вместо `src`
- Model Registry в MLflow

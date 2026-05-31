Project Structure
=================

Где лежит код и куда добавлять новые части пайплайна.
План рефакторинга: ``docs/refactoring-plan.md``.

Основные директории
-------------------

``configs/``
   YAML-конфиги. Production: ``xgboost_baseline.yaml``.
   Эксперименты: ``configs/experiments/*.yaml``.

``data/{raw,interim,processed,external}/``
   Данные. Train читает ``data/processed/fin_data.csv``.

``models/xgboost/``
   Production-артефакты после ``make train``.

``reports/figures/``
   Confusion matrix после ``make train``.

``reports/figures/experiments/<run_name>/``
   Отчёты по каждому MLflow-run (не коммитятся).

``notebooks/``
   EDA; логику пайплайна переносить в ``src/``.

``tests/``
   pytest: features, preprocess, API.

``docs/``
   Sphinx (``.rst``) и MD-гайды.

Модули ``src``
--------------

``src/config.py``
   Пути, dotenv, MLflow/MinIO env, имена колонок.

``src/config_loader.py``
   ``ExperimentConfig`` из YAML.

``src/cli.py``
   Entry point ``study-mlops``.

Data
^^^^

``src/data/make_dataset.py`` — копирование CSV в ``processed/``.
``src/data/loaders.py`` — загрузка + валидация.
``src/data/validation.py`` — схема датасета.

Features
^^^^^^^^

``src/features/preprocess.py`` — очистка, stemming.
``src/features/numerical.py`` — числовые признаки (train = predict).
``src/features/build_features.py`` — TF-IDF, SVD, ``transform_single``.

Models
^^^^^^

``src/models/trainer.py`` — обучение, метрики, отчёты.
``src/models/registry.py`` — ``save_artifacts``, ``MentalHealthPredictor``.
``src/models/tracking.py`` — MLflow (метрики, модель, ``evaluation/``).
``src/models/factory.py`` — ``MODEL_REGISTRY``.
``src/models/xgboost_model.py``, ``lightgbm_model.py``, ``sklearn_models.py``.
``src/models/run_experiments.py`` — пакетный запуск конфигов.
``src/models/train_model.py``, ``predict_model.py`` — Click CLI.

API и визуализация
^^^^^^^^^^^^^^^^^^

``src/api/`` — FastAPI ``/predict``, ``/health``.
``src/visualization/visualize.py`` — confusion matrix, classification report.

Поддерживаемые модели (``model.name`` в YAML)
---------------------------------------------

* ``xgboost``
* ``lightgbm`` (нужен ``pip install lightgbm``)
* ``logistic_regression``
* ``random_forest``

Правила
-------

* Гиперпараметры — в ``configs/``, не в коде.
* Новая модель — builder в ``src/models/`` + запись в ``MODEL_REGISTRY``.
* CSV, ``.pkl``, ``models/`` — не в Git (см. ``.gitignore``).

Getting started
===============

Проект предназначен для классификации текстов по категориям mental health.
Production-код — в ``src/``, исследования — в ``notebooks/``.

Установка
---------

Основной файл окружения: ``environment.yml``.

.. code-block:: bash

   conda env update --name ML_Ops --file environment.yml --prune
   conda activate ML_Ops
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

Для экспериментов с LightGBM (если пакет не в conda-окружении):

.. code-block:: bash

   pip install lightgbm

Скопируйте ``.env.example`` в ``.env`` для MLflow/MinIO (опционально; без
файла используются локальные дефолты ``minioadmin``).

Подготовка данных
-----------------

.. code-block:: bash

   make data
   dvc pull    # если данные в DVC remote

Ожидаемый файл: ``data/processed/fin_data.csv``.

Обучение (production)
---------------------

Базовый конфиг: ``configs/xgboost_baseline.yaml``. Локальные артефакты
перезаписывают ``models/xgboost/``.

.. code-block:: bash

   make infra-up    # опционально: MLflow :5000, MinIO :9001
   make train

Пакетные эксперименты (MLflow)
--------------------------------

Запускает все YAML из ``configs/experiments/`` (XGBoost, LightGBM,
Logistic Regression, Random Forest). Локальную production-модель не
перезаписывает. В MLflow — метрики, модель и ``evaluation/confusion_matrix.png``.

.. code-block:: bash

   make experiments
   # или
   study-mlops experiments

Один эксперимент:

.. code-block:: bash

   python -m src.models.train_model \
     --config configs/experiments/lightgbm_baseline.yaml

Инференс
--------

CLI (модель из ``models/xgboost/``):

.. code-block:: bash

   make predict TEXT="I feel sad and anxious and cannot sleep."

HTTP API:

.. code-block:: bash

   make serve
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I feel sad and anxious."}'

   curl http://localhost:8000/health

Тесты и качество кода
---------------------

.. code-block:: bash

   make test
   make lint
   tox

MLOps
-----

.. code-block:: bash

   make dvc-repro
   make dvc-pull

См. также ``docs/mlflow-minio-setup.md`` и ``docs/refactoring-plan.md``.

Commands
========

Точки входа: **Makefile**, **study-mlops** CLI и модули ``python -m src.*``.

Core commands
^^^^^^^^^^^^^

* ``make environment`` — создать/обновить conda-окружение ``ML_Ops`` из ``environment.yml``.
* ``make requirements`` — ``pip install -r requirements.txt``.
* ``make data`` — скопировать ``fin_data.csv`` в ``data/processed/``.
* ``make train`` — обучение по ``configs/xgboost_baseline.yaml``, сохранение в ``models/xgboost/``.
* ``make experiments`` — все конфиги из ``configs/experiments/``, логирование в MLflow.
* ``make predict TEXT="..."`` — инференс по артефактам ``models/xgboost/``.
* ``make test`` — ``pytest tests/``.
* ``make serve`` — FastAPI на порту 8000.
* ``make lint`` — ``flake8 src``.
* ``make clean`` — удалить ``__pycache__`` и ``*.pyc``.

CLI (``pip install -e .``)
^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``study-mlops data [input] [output]`` — подготовка данных.
* ``study-mlops train [--config PATH]`` — обучение.
* ``study-mlops experiments [--configs-dir DIR]`` — пакет экспериментов.
* ``study-mlops predict "text"`` — инференс.
* ``study-mlops infra status`` — статус Docker Compose.

Модули Python
^^^^^^^^^^^^^

* ``python -m src.data.make_dataset data/raw data/processed``
* ``python -m src.models.train_model --config configs/xgboost_baseline.yaml``
* ``python -m src.models.run_experiments --configs-dir configs/experiments``
* ``python -m src.models.predict_model "your text here"``

Infrastructure and DVC
^^^^^^^^^^^^^^^^^^^^^^

* ``make infra-up`` / ``infra-down`` / ``infra-status`` / ``infra-logs`` — MinIO + MLflow.
* ``make dvc-repro`` — DVC pipeline: ``prepare`` → ``train``.
* ``make dvc-pull`` — скачать данные из DVC remote.

Переменные окружения: ``.env.example`` → ``.env``. Подробнее:
``docs/mlflow-minio-setup.md``.

Syncing data to S3 (AWS CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``make sync_data_to_s3`` — ``aws s3 sync data/`` в bucket.
* ``make sync_data_from_s3`` — sync из bucket в ``data/``.

Требуется ``awscli`` (не в ``requirements.txt``).

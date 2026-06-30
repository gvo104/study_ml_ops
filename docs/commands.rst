Commands
========

Основные сценарии запуска сосредоточены в ``Makefile`` и CLI ``study-mlops``.

Makefile
--------

Базовые команды:

* ``make data`` — подготовить ``data/processed/fin_data.csv``.
* ``make train`` — обучить baseline-модель по ``configs/xgboost_baseline.yaml``.
* ``make experiments`` — запустить experiment-конфиги из ``configs/experiments/``.
* ``make predict TEXT="..."`` — выполнить CLI-инференс.
* ``make serve`` — поднять основной FastAPI-сервис на порту 8000.
* ``make serve-synthetic`` — поднять synthetic API на порту 8001.
* ``make test`` — запустить основной pytest suite.
* ``make test-synthetic`` — запустить только synthetic API тесты.
* ``make lint`` — проверить ``src`` через flake8.

Инфраструктура:

* ``make infra-up`` — поднять MinIO, MLflow и webapp через ``docker compose``.
* ``make infra-down`` — остановить локальную инфраструктуру.
* ``make infra-status`` — показать статус compose-сервисов.
* ``make infra-logs`` — показать логи MLflow.

DVC:

* ``make dvc-repro`` — воспроизвести DVC pipeline.
* ``make dvc-pull`` — скачать данные из DVC remote.

Monitoring:

* ``make drift-monitoring-offline`` — offline playback и monitoring stack.
* ``make drift-monitoring-online`` — online synthetic loop и monitoring stack.
* ``make monitoring-status`` — статус Prometheus/Grafana/exporter.
* ``make monitoring-down`` — остановить monitoring stack.

CLI ``study-mlops``
-------------------

Доступные команды:

* ``study-mlops data [input] [output]`` — подготовка processed dataset.
* ``study-mlops train --data-path ... --config ...`` — обучение модели.
* ``study-mlops experiments --data-path ... --configs-dir ...`` — пакет экспериментов.
* ``study-mlops predict "text"`` — инференс по текущим артефактам модели.
* ``study-mlops infra status`` — статус Docker Compose инфраструктуры.

Прямые Python entrypoints
-------------------------

* ``python -m src.data.make_dataset data/raw data/processed``
* ``python -m src.models.train_model --data-path data/processed/fin_data.csv --config configs/xgboost_baseline.yaml``
* ``python -m src.models.run_experiments --data-path data/processed/fin_data.csv --configs-dir configs/experiments``
* ``python -m src.models.predict_model "your text here"``
* ``python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000``

Примечания
----------

- Все команды в документации предполагают активированное окружение
  ``conda activate ML_Ops``.
- Команды ``infra-*`` и monitoring-команды требуют Docker Compose или
  совместимый runtime.
- ``sync_data_to_s3`` и ``sync_data_from_s3`` существуют в ``Makefile``, но
  требуют отдельно установленный ``awscli`` и не относятся к базовому
  сценарию работы с проектом.

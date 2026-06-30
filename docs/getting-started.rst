Getting started
===============

Эта страница описывает базовый сценарий работы с проектом в окружении
``ML_Ops``.

Установка окружения
-------------------

.. code-block:: bash

   conda env update --name ML_Ops --file environment.yml --prune
   conda activate ML_Ops
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

При необходимости можно дополнительно проверить интерпретатор:

.. code-block:: bash

   python --version

Подготовка данных
-----------------

Основной dataset ожидается в ``data/processed/fin_data.csv``.

Подготовить processed dataset:

.. code-block:: bash

   conda activate ML_Ops
   make data

Или подтянуть данные через DVC:

.. code-block:: bash

   conda activate ML_Ops
   make dvc-pull

Baseline-обучение
-----------------

Базовый конфиг — ``configs/xgboost_baseline.yaml``.

.. code-block:: bash

   conda activate ML_Ops
   make train

После обучения артефакты сохраняются в ``models/xgboost/``.

Инференс
--------

CLI:

.. code-block:: bash

   conda activate ML_Ops
   make predict TEXT="I feel sad and anxious and cannot sleep."

HTTP API:

.. code-block:: bash

   conda activate ML_Ops
   make serve

После запуска доступны:

- ``http://localhost:8000/``
- ``http://localhost:8000/experiments``
- ``http://localhost:8000/docs``
- ``http://localhost:8000/health``

Пример запроса:

.. code-block:: bash

   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I feel sad and anxious."}'

MLflow и MinIO
--------------

Локальная инфраструктура поднимается через Docker Compose:

.. code-block:: bash

   conda activate ML_Ops
   cp .env.example .env
   make infra-up

После запуска:

- MLflow UI: ``http://localhost:5000``
- MinIO Console: ``http://localhost:9001``

Эксперименты
------------

Все YAML-конфиги из ``configs/experiments/``:

.. code-block:: bash

   conda activate ML_Ops
   make experiments

CLI-эквивалент:

.. code-block:: bash

   conda activate ML_Ops
   study-mlops experiments --data-path data/processed/fin_data.csv --configs-dir configs/experiments

Drift monitoring
----------------

Офлайн demo monitoring:

.. code-block:: bash

   conda activate ML_Ops
   make drift-monitoring-offline

Online synthetic monitoring:

.. code-block:: bash

   conda activate ML_Ops
   make drift-monitoring-online

Отдельный synthetic API:

.. code-block:: bash

   conda activate ML_Ops
   make serve-synthetic

Проверки
--------

.. code-block:: bash

   conda activate ML_Ops
   make test
   make lint

Связанные документы
-------------------

- ``README.md`` — полный обзор проекта
- ``commands.rst`` — список основных команд
- ``project-structure.rst`` — структура модулей и каталогов

Getting started
===============

Проект предназначен для классификации текстов по категориям mental health.
Основной рабочий код лежит в ``src/`` внутри шаблонной папки ``study_ml_ops/``.

Установка
---------

Основной файл окружения проекта: ``environment.yml``.

Из папки ``study_ml_ops/`` создайте или обновите conda-окружение:

.. code-block:: bash

   conda env update --name ML_Ops --file environment.yml --prune
   conda activate ML_Ops

Если conda не используется, можно поставить pip-зависимости из дубликата
``requirements.txt``:

.. code-block:: bash

   make requirements

Подготовка данных
-----------------

Пайплайн обучения ожидает файл:

.. code-block:: text

   data/processed/fin_data.csv

Если исходный CSV лежит в ``data/raw/fin_data.csv``, выполните:

.. code-block:: bash

   make data

Если ``data/processed/fin_data.csv`` уже существует, ``make data`` завершится
без переустановки зависимостей. Если ``data/raw/fin_data.csv`` еще нет,
команда попробует взять старую локальную копию из ``../data/fin_data.csv``.

Если CSV лежит в другом месте, скопируйте его так:

.. code-block:: bash

   python -m src.data.make_dataset ../data/fin_data.csv data/processed

Обучение
--------

.. code-block:: bash

   make train

Модель и связанные артефакты сохраняются в ``models/xgboost/``.

Инференс
--------

.. code-block:: bash

   make predict TEXT="I feel sad and anxious and cannot sleep."

Перед инференсом в ``models/xgboost/`` должны лежать сохраненные артефакты
обученной модели.

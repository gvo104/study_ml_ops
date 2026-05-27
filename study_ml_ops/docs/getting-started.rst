Getting started
===============

Проект предназначен для классификации текстов по категориям mental health.
Основной рабочий код лежит в ``src/`` внутри шаблонной папки ``study_ml_ops/``.

Установка
---------

Из папки ``study_ml_ops/`` установите зависимости:

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

Если CSV лежит в старой корневой структуре проекта, например
``../data/fin_data.csv``, скопируйте его так:

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

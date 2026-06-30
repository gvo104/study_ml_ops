Project structure
=================

Этот документ дополняет корневой ``README.md`` и кратко описывает основные
подсистемы репозитория.

Основной Python-пакет ``src/``
------------------------------

``src.api``
   FastAPI-приложение, встроенный web UI, Pydantic-схемы, dashboard state и
   bootstrap логика модели/датасета.

``src.data``
   Подготовка ``data/processed/fin_data.csv``, загрузка датасета и его
   валидация.

``src.features``
   Текстовая предобработка, числовые признаки и сборка признакового пайплайна
   для обучения и инференса.

``src.models``
   Обучение, пакетные эксперименты, CLI-инференс, MLflow tracking, локальные
   артефакты модели и реестр поддерживаемых моделей.

``src.visualization``
   Построение confusion matrix и сохранение classification report.

``src.config`` / ``src.config_loader`` / ``src.cli``
   Пути, env-настройки, YAML-конфиги и главная CLI-группа ``study-mlops``.

Подсистема ``drift_v2/``
------------------------

``drift_v2.runner``
   Оконные drift-метрики, orchestration offline/online run-ов, запись
   отчетов и JSON-артефактов.

``drift_v2.monitoring``
   Prometheus exporter и чтение последних monitoring snapshots.

``drift_v2.synthetic_api``
   Отдельный FastAPI-сервис для synthetic generator/expert сценариев.

``drift_v2.store``
   SQLite schema и доступ к event store.

``drift_v2.reference`` / ``drift_v2.configs``
   Reference statistics и конфиги monitoring-пайплайна.

Инфраструктурные каталоги
-------------------------

``configs/``
   YAML-конфиги baseline-модели и experiment runs.

``monitoring/``
   Конфиги Prometheus и Grafana dashboards для ``drift_v2``.

``k8s/``
   Kubernetes manifests для MinIO и MLflow.

``arcd/``
   Argo CD application manifest для инфраструктурных манифестов ``k8s/base``.

``.github/workflows/``
   GitHub Actions workflows для lint/test/build/deploy сценариев.

``tests/``
   Тесты основного API, features, numerical/preprocess логики и подсистемы
   ``drift_v2``.

``docs/``
   Sphinx-документация верхнего уровня.

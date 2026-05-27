Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Core commands
^^^^^^^^^^^^^

* ``make environment`` creates or updates the ``ML_Ops`` conda environment from ``environment.yml``.
* ``make requirements`` installs pip dependencies from ``requirements.txt``.
* ``make data`` copies ``data/raw/fin_data.csv`` into ``data/processed/fin_data.csv``.
* ``make train`` trains the XGBoost text classifier on ``data/processed/fin_data.csv``.
* ``make predict TEXT="..."`` runs inference with artifacts from ``models/xgboost/``.
* ``make lint`` runs ``flake8 src``.
* ``make clean`` removes compiled Python files and ``__pycache__`` directories.

``make data`` does not install dependencies. If ``data/processed/fin_data.csv``
already exists, it exits successfully without copying anything. If
``data/raw/fin_data.csv`` is missing, it tries to import the legacy local copy
from ``../data/fin_data.csv``.

Syncing data to S3
^^^^^^^^^^^^^^^^^^

* ``make sync_data_to_s3`` uses ``aws s3 sync`` to sync ``data/`` to the configured bucket.
* ``make sync_data_from_s3`` uses ``aws s3 sync`` to sync the configured bucket into ``data/``.

These commands require ``awscli`` in the active environment. It is intentionally
not installed by ``requirements.txt`` to avoid slow pip dependency resolution.

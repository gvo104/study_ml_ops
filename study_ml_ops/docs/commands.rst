Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Core commands
^^^^^^^^^^^^^

* ``make requirements`` installs project dependencies from ``requirements.txt``.
* ``make data`` copies ``data/raw/fin_data.csv`` into ``data/processed/fin_data.csv``.
* ``make train`` trains the XGBoost text classifier on ``data/processed/fin_data.csv``.
* ``make predict TEXT="..."`` runs inference with artifacts from ``models/xgboost/``.
* ``make lint`` runs ``flake8 src``.
* ``make clean`` removes compiled Python files and ``__pycache__`` directories.

Syncing data to S3
^^^^^^^^^^^^^^^^^^

* `make sync_data_to_s3` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://[OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')/data/`.
* `make sync_data_from_s3` will use `aws s3 sync` to recursively sync files from `s3://[OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')/data/` to `data/`.

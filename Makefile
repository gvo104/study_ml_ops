.PHONY: clean data environment lint requirements train predict test test-synthetic test-drift test-monitoring serve serve-synthetic serve-drift-exporter run-drift-debug run-drift-full experiments dvc-repro dvc-pull sync_data_to_s3 sync_data_from_s3 monitoring-up monitoring-down monitoring-status

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = study_ml_ops
PYTHON_INTERPRETER = python3
ENV_NAME = ML_Ops

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies with pip
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Create or update conda environment from environment.yml
environment:
	conda env update --name $(ENV_NAME) --file environment.yml --prune

## Make Dataset
data:
	$(PYTHON_INTERPRETER) -m src.data.make_dataset data/raw data/processed

## Train XGBoost text classifier
train:
	$(PYTHON_INTERPRETER) -m src.models.train_model --data-path data/processed/fin_data.csv --config configs/xgboost_baseline.yaml

## Run all MLflow experiments (configs/experiments/*.yaml)
experiments:
	$(PYTHON_INTERPRETER) -m src.models.run_experiments --data-path data/processed/fin_data.csv

## Run pytest suite
test:
	$(PYTHON_INTERPRETER) -m pytest tests/ -q

## Run only synthetic-api tests
test-synthetic:
	$(PYTHON_INTERPRETER) -m pytest tests/test_synthetic_api.py -q

## Run only drift-runner tests
test-drift:
	$(PYTHON_INTERPRETER) -m pytest tests/test_drift_runner.py -q

## Run only drift monitoring exporter tests
test-monitoring:
	$(PYTHON_INTERPRETER) -m pytest tests/test_drift_monitoring.py -q

## Start FastAPI inference server
serve:
	$(PYTHON_INTERPRETER) -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

## Start synthetic FastAPI service for generator/expert roles
serve-synthetic:
	$(PYTHON_INTERPRETER) -m uvicorn drift.synthetic_api.app:app --reload --host $${SYNTHETIC_API_HOST:-0.0.0.0} --port $${SYNTHETIC_API_PORT:-8001}

## Start drift Prometheus exporter locally
serve-drift-exporter:
	$(PYTHON_INTERPRETER) -m uvicorn drift.monitoring.app:app --reload --host $${DRIFT_EXPORTER_HOST:-0.0.0.0} --port $${DRIFT_EXPORTER_PORT:-9108}

## Run drift-runner in debug mode
run-drift-debug:
	$(PYTHON_INTERPRETER) -m drift.runner.cli run --config drift/configs/runner.yaml --mode debug

## Run drift-runner in full mode
run-drift-full:
	$(PYTHON_INTERPRETER) -m drift.runner.cli run --config drift/configs/runner.yaml --mode full

## Reproduce DVC pipeline (prepare + train)
dvc-repro:
	dvc repro

## Pull datasets from DVC remote
dvc-pull:
	dvc pull

## Predict mental health class for TEXT='your text'
predict:
	$(PYTHON_INTERPRETER) -m src.models.predict_model "$(TEXT)"

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3, requires awscli in the active environment
sync_data_to_s3:
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)

## Download Data from S3, requires awscli in the active environment
sync_data_from_s3:
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

#################################################################################
# INFRASTRUCTURE (Docker Compose: MinIO + MLflow)                               #
#################################################################################

## Start MinIO (S3) + MLflow tracking server
infra-up:
	docker compose up -d
	@echo "============================================"
	@echo "Infrastructure started!"
	@echo "MinIO Console: http://localhost:9001"
	@echo "MLflow UI:     http://localhost:5000"
	@echo "MinIO credentials: minioadmin / minioadmin"
	@echo "============================================"

## Stop all infrastructure containers
infra-down:
	docker compose down

## Show status of infrastructure containers
infra-status:
	docker compose ps

## View MLflow logs
infra-logs:
	docker compose logs -f mlflow

## Start Prometheus + Grafana + drift exporter
monitoring-up:
	docker compose -f docker-compose.monitoring.yml up -d

## Stop Prometheus + Grafana + drift exporter
monitoring-down:
	docker compose -f docker-compose.monitoring.yml down

## Show monitoring stack container status
monitoring-status:
	docker compose -f docker-compose.monitoring.yml ps

.PHONY: clean data environment lint requirements train predict test serve experiments dvc-repro dvc-pull sync_data_to_s3 sync_data_from_s3 k8s-forward argocd-install argocd-app argocd-password argocd-sync

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

## Start FastAPI inference server
serve:
	$(PYTHON_INTERPRETER) -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

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

## Forward Kubernetes services to localhost:9000, localhost:5000, localhost:8000 and Argo CD to localhost:8080
k8s-forward:
	./scripts/k8s-port-forward.sh

## Install Argo CD into the current Kubernetes cluster
argocd-install:
	kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply --server-side --force-conflicts -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

## Register the ml-team application in Argo CD
argocd-app:
	kubectl apply -k argocd/

## Print the initial Argo CD admin password
argocd-password:
	kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath='{.data.password}' | base64 -d
	@echo

## Force sync and wait for the ml-team application
argocd-sync:
	argocd app sync ml-team-app
	argocd app wait ml-team-app --health --sync --timeout 300

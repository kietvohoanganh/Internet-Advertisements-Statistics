RSCRIPT ?= Rscript
PYTHON ?= .venv/bin/python
PYTHON_BOOTSTRAP ?= python3.11
OUTPUT ?= predictions.csv

.PHONY: setup download run predict predict-legacy publish-results test check

setup:
	"$(RSCRIPT)" scripts/setup.R
	"$(PYTHON_BOOTSTRAP)" -m venv .venv
	"$(PYTHON)" -m pip install -r requirements.txt

download:
	"$(RSCRIPT)" scripts/download.R

run:
	PYTHON="$(PYTHON)" "$(RSCRIPT)" scripts/run.R

predict:
	@test -n "$(INPUT)" || { echo 'Set INPUT to an RDS data frame of UCI features'; exit 2; }
	PYTHON="$(PYTHON)" "$(RSCRIPT)" scripts/predict.R "$(INPUT)" "$(OUTPUT)"

predict-legacy:
	@set -eu; test -n "$(INPUT)" || { echo 'Set INPUT to an RDS data frame of UCI features'; exit 2; }; \
	stage=$$(mktemp -d); trap 'rm -rf "$$stage"' EXIT; \
	"$(RSCRIPT)" classifier/evidence.R "$(INPUT)" "$$stage"; \
	"$(PYTHON)" classifier/correction.py predict "$$stage" "$(OUTPUT)"

test:
	"$(RSCRIPT)" tests/smoke_test.R
	"$(PYTHON)" tests/classifier_test.py
	"$(PYTHON)" tests/benchmark_test.py
	"$(RSCRIPT)" tests/solver_test.R
	"$(PYTHON)" tests/publish_test.py

check: test
	"$(RSCRIPT)" tests/integration_test.R
	PYTHON="$(PYTHON)" "$(RSCRIPT)" tests/benchmark_integration.R
	"$(RSCRIPT)" tests/validate_outputs.R
	"$(PYTHON)" scripts/publish_results.py --check

publish-results:
	RSCRIPT="$(RSCRIPT)" "$(PYTHON)" scripts/publish_results.py

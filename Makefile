# PoseML — reproducible conversion toolchain.
# All Python runs go through `uv` against the project-local .venv (fully self-contained).

UV := uv
RUN := $(UV) run

.PHONY: help setup models inspect lint test clean

help:
	@echo "PoseML targets:"
	@echo "  setup    - create/sync the project venv (all dependency groups)"
	@echo "  models   - download the BlazePose .tflite models"
	@echo "  inspect  - dump each model's I/O contract -> models/tflite/io_contract.json"
	@echo "  verify   - parity test: PyTorch port vs every .tflite reference"
	@echo "  parity   - detailed parity report for one MODEL=<path>"
	@echo "  lint     - ruff check"
	@echo "  test     - pytest"
	@echo "  clean    - remove downloaded model binaries"

setup:
	$(UV) sync

models:
	$(RUN) python -m poseml.convert.download_models

inspect:
	$(RUN) --group reference python -m poseml.convert.inspect_tflite

# Numerical parity of the PyTorch port vs each .tflite reference.
verify:
	$(RUN) --group reference pytest

# Detailed per-output parity report for one model (MODEL=path).
MODEL ?= models/tflite/pose_landmark_full.tflite
parity:
	$(RUN) --group reference python -m poseml.verify.parity $(MODEL)

lint:
	$(RUN) ruff check python

test:
	$(RUN) pytest

clean:
	rm -f models/tflite/*.tflite models/tflite/io_contract.json

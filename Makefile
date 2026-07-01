# PoseML — reproducible conversion toolchain.
# All Python runs go through `uv` against the project-local .venv (fully self-contained).

UV := uv
RUN := $(UV) run

.PHONY: help setup models inspect lint test clean coreml coreml-verify

# The two models we ship first: full-frame detector + balanced landmark variant.
COREML_MODELS := pose_detection pose_landmark_full

help:
	@echo "PoseML targets:"
	@echo "  setup         - create/sync the project venv (all dependency groups)"
	@echo "  models        - download the BlazePose .tflite models"
	@echo "  inspect       - dump each model's I/O contract -> models/tflite/io_contract.json"
	@echo "  verify        - parity test: PyTorch port vs every .tflite reference"
	@echo "  parity        - detailed parity report for one MODEL=<path>"
	@echo "  coreml        - convert detector + landmark(full) to .mlpackage (fp16, ANE)"
	@echo "  coreml-verify - Core ML vs tflite parity (fp16 budget + strict fp32 fidelity)"
	@echo "  lint          - ruff check"
	@echo "  test          - pytest"
	@echo "  clean         - remove downloaded model binaries"

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

# Convert the shipping models to Core ML ML Programs (fp16, ImageType input, mask/heatmap pruned).
coreml:
	@for m in $(COREML_MODELS); do \
		$(RUN) --group reference python -m poseml.convert.to_coreml \
			--model models/tflite/$$m.tflite || exit 1; \
	done

# Core ML vs tflite parity. fp16 = shipping-model precision budget; fp32 = strict conversion fidelity.
coreml-verify:
	@for m in $(COREML_MODELS); do \
		echo "=== $$m (fp16, shipping) ==="; \
		$(RUN) --group reference python -m poseml.verify.coreml_parity \
			--model models/tflite/$$m.tflite --coreml models/coreml/$$m.mlpackage || exit 1; \
		echo "=== $$m (fp32, conversion fidelity) ==="; \
		$(RUN) --group reference python -m poseml.convert.to_coreml \
			--model models/tflite/$$m.tflite --precision fp32 || exit 1; \
		$(RUN) --group reference python -m poseml.verify.coreml_parity \
			--model models/tflite/$$m.tflite --coreml models/coreml/$$m.fp32.mlpackage \
			--atol 5e-3 --rtol 5e-3 || exit 1; \
		rm -rf models/coreml/$$m.fp32.mlpackage; \
	done

lint:
	$(RUN) ruff check python

test:
	$(RUN) pytest

clean:
	rm -f models/tflite/*.tflite models/tflite/io_contract.json

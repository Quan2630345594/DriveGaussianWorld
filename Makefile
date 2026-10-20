PYTHON ?= python
DEVICE ?= cuda
CONFIG ?= configs/default.json

.PHONY: test doctor pipeline

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q -p no:cacheprovider

doctor:
	$(PYTHON) -m drive_gaussian_world doctor

pipeline:
	DGW_PYTHON=$(PYTHON) DGW_DEVICE=$(DEVICE) DGW_CONFIG=$(CONFIG) bash scripts/pipeline.sh

PYTHON ?= python
export CUDA_VISIBLE_DEVICES :=
export PYTHONNOUSERSITE := 1
export OMP_NUM_THREADS := 2
export OPENBLAS_NUM_THREADS := 2
export PYTEST_DISABLE_PLUGIN_AUTOLOAD := 1

.PHONY: test doctor demo pipeline

test:
	$(PYTHON) -m pytest -q -p no:cacheprovider

doctor:
	$(PYTHON) -m drive_gaussian_world doctor

pipeline:
	DGW_PYTHON=$(PYTHON) bash scripts/cpu_pipeline.sh

demo:
	$(PYTHON) -m drive_gaussian_world demo --dataset data/processed/kitti_dggt --checkpoint runs/driving_cpu/dggt_world/best.pt --output runs/demo_$(shell date +%s)

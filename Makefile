PYTHON ?= python
MAMBA_ENV ?= lab-realtime-stt
URL ?= http://127.0.0.1:7860

.PHONY: setup update run test smoke clean

setup:
	mamba env create -f environment-lab-realtime-stt.yml

update:
	mamba env update -n $(MAMBA_ENV) -f environment-lab-realtime-stt.yml

run:
	mamba run -n $(MAMBA_ENV) ./scripts/run_server.sh

test:
	mamba run -n $(MAMBA_ENV) $(PYTHON) -m pytest -q

smoke:
	mamba run -n $(MAMBA_ENV) $(PYTHON) scripts/smoke_check.py --url $(URL)

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +

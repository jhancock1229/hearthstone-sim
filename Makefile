.PHONY: test lint clean sim sim-1k sim-10k sim-verbose sim-save

test:
	python3 -m pytest tests/ -v

test-cov:
	python3 -m pytest tests/ -v --cov=hearthstone --cov=simulation

lint:
	ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

# Simulation targets
sim:
	python3 scripts/simulate.py --games 100 --seed 42

sim-1k:
	python3 scripts/simulate.py --games 1000 --seed 42

sim-10k:
	python3 scripts/simulate.py --games 10000 --seed 42 --verbose

sim-verbose:
	python3 scripts/simulate.py --games 1000 --seed 42 --verbose

sim-save:
	python3 scripts/simulate.py --games 1000 --seed 42 --output simulation_results.json
	@echo "Results saved to simulation_results.json"

.PHONY: demo test failrate attack plots
demo:
	python -m lab.demo

test:
	python -m pytest -q

failrate:
	python -m experiments.failrate

attack:
	python -m experiments.scale

plots:
	python -m experiments.plots

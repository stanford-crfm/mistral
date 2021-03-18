SHELL := /bin/bash
.PHONY: help serialize-env check autoformat prune
.DEFAULT: help

# Create Valid Architectures
ARCHITECTURES := cpu gpu

# Generates a useful overview/help message for various make features - add to this as necessary!
help:
	@echo "make serialize-env arch=<ID>"
	@echo "    After (un)installing dependencies, dump environment.yaml for arch :: < cpu | gpu >."
	@echo "make prune"
	@echo "    Pull all branches from git, and prune all local branches that are merged in origin."
	@echo "make check"
	@echo "    Run code style and linting (black, flake, isort) *without* changing files!"
	@echo "make autoformat"
	@echo "    Run code styling (black, isort) and update in place - committing with pre-commit also does this."

serialize-env:
ifneq ($(filter $(arch),$(ARCHITECTURES)),)
	python environments/export.py -a $(arch)
else
	@echo "Argument 'arch' is not set - try calling 'make serialize-env arch=<ID>' with ID = < cpu | gpu >."
endif

check:
	isort --check .
	black --check .
	flake8 .

autoformat:
	isort --atomic .
	black .

prune:
	@bash -c "git fetch -p";
	@bash -c "for branch in $(git branch -vv | grep ': gone]' | awk '{print $1}'); do git branch -d $branch; done";

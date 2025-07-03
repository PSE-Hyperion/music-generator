#!/bin/sh

# Install virtual environment
python -m venv ../.venv
../.venv/bin/pip install -U pip
../.venv/bin/pip install -r ../requirements.txt
PYTHONPATH="${HOME}/music-generator/.venv/bin/python:${PYTHONPATH}"

# git configuration
git config core.autocrlf false


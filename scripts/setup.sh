#!/bin/sh
PROJECTPATH="${HOME}/music-generator"

# Install virtual environment
python -m venv ${PROJECTPATH}/.venv
${PROJECTPATH}/.venv/bin/pip install -U pip
${PROJECTPATH}/.venv/bin/pip install -r ../requirements.txt

# git configuration
git config core.autocrlf false


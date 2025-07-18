#!/bin/bash
PROJECTPATH="${HOME}/music-generator"

# Install virtual environment
python3.11 -m venv --clear ${PROJECTPATH}/.venv
${PROJECTPATH}/.venv/bin/pip install -U pip
${PROJECTPATH}/.venv/bin/pip install -r ${PROJECTPATH}/requirements.txt

# configs
git config core.autocrlf false

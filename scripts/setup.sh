#!/bin/bash
PROJECTPATH="${HOME}/Coding/music-generator"

# Install virtual environment
python3.11 -m --clear venv ${PROJECTPATH}/.venv
${PROJECTPATH}/.venv/bin/pip install -U pip
${PROJECTPATH}/.venv/bin/pip install -r ${PROJECTPATH}/requirements.txt

# configs
git config core.autocrlf false

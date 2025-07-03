#!/bin/bash
PROJECTPATH="${HOME}/music-generator"

# Install python system packages
sudo dnf install python3.11 python3.11-devel

# Install virtual environment
python3.11 -m venv ${PROJECTPATH}/.venv
${PROJECTPATH}/.venv/bin/pip install -U pip
${PROJECTPATH}/.venv/bin/pip install -r ../requirements.txt



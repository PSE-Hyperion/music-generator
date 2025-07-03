#!/bin/bash
sudo dnf upgrade
sudo dnf install python3.11 python3.11-devel docker

# Configs
git config core.autocrlf false

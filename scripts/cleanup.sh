#!/bin/bash

docker system prune -a --volumes
rm -r ../.venv
rm -r ../.cache

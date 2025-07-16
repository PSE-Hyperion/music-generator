#!/bin/bash

DOCKER_BUILDKIT=1
CONFIG_PATH="${HOME}/music-generator/.docker/docker-compose.yml"
SERVICE_NAME="app"

docker compose -f ${CONFIG_PATH} run --rm --build ${SERVICE_NAME}

#!/bin/bash

CONFIG_PATH = "../.docker/docker-compose.yml"
SERVICE_NAME = "app"

docker compose -f ${CONFIG_PATH} --rm --build ${SERVICE_NAME}

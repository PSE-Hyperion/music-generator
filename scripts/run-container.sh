#!/bin/bash

DOCKER_BUILDKIT=1
PROJECT_PATH="${HOME}/music-generator"
DOCKER_COMPOSE_PATH="${PROJECT_PATH}/.docker/docker-compose.yml"
DOCKERFILE_GPU_PATH="${PROJECT_PATH}/.docker/Dockerfile.gpu"

if [[ "$1" != "--gpu" ]]; then
    echo "Run default docker program"
    docker compose -f ${DOCKER_COMPOSE_PATH} run --build --rm "app"
else
    echo "Run docker program with GPU"
    shift
    docker build -f ${DOCKERFILE_GPU_PATH} -t "music-gen:gpu" ${PROJECT_PATH}
    docker run --gpus all -u 1000:1000 --rm -it music-gen:gpu
fi

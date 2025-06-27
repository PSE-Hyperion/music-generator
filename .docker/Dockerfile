# Set the base image for the project image
ARG	    TAG=latest
FROM 	tensorflow/tensorflow:${TAG} AS base

# Stage 1: Building the image
FROM	base AS builder

RUN 	apt-get update && apt-get install -y --no-install-recommends \
      	build-essential libxml2-dev libfreetype6-dev libpng-dev \
    	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY	../pyproject.toml ./

RUN 	pip install -U --no-cache-dir pip \
        && pip install --no-cache-dir ./

# Staeg 2: Create clean image
FROM 	base

WORKDIR	/app

COPY	--from=builder /usr/local /usr/local

WORKDIR	/app

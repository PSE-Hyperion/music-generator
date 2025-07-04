# Set the base image for the project image
ARG	    EDITION=latest
FROM 	tensorflow/tensorflow:${EDITION} AS base

# Stage 1: Building the image
FROM	base AS builder

RUN 	apt-get update && apt-get install -y --no-install-recommends \
      	build-essential libxml2-dev libfreetype6-dev libpng-dev \
    	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY	./ ./

RUN 	--mount=type=cache,target=/root/.cache/pip \
        pip install -U pip && \
        pip install -r requirements.txt && \
        pip install ./

# Staeg 2: Create clean image
FROM 	base

WORKDIR	/app

COPY	--from=builder /usr/local /usr/local

WORKDIR	/app

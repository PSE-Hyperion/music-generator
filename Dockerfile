ARG	TAG=latest-jupyter
FROM 	tensorflow/tensorflow:${TAG} AS builder

RUN 	apt-get update && apt-get install -y --no-install-recommends \
      	build-essential libxml2-dev libfreetype6-dev libpng-dev git \
    	&& rm -rf /var/lib/apt/lists/*

RUN 	pip install --upgrade --no-cache-dir \
	  pip music21 matplotlib ruff

WORKDIR /app
CMD 	["python"]

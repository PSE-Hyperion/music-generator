#!/usr/bin/python
import subprocess
import sys

CONFIG_PATH = ".docker/docker-compose.yml"
SERVICE_NAME = "app"

subprocess.check_call(["docker", "compose", "-f", CONFIG_PATH, "run", "--rm", "--build", SERVICE_NAME] + sys.argv[1:])

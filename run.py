#!/usr/bin/python
import os
import subprocess
import sys

IMAGE_NAME = "music_generator"
DOCKERFILE_PATH = "."

subprocess.check_call(
        ["docker", "build", "-t",
         IMAGE_NAME, DOCKERFILE_PATH]
        )
subprocess.check_call(
        ["docker", "run", "-it", "--rm", "-v",
         f"{os.getcwd()}:/app", IMAGE_NAME] + sys.argv[1:]
        )

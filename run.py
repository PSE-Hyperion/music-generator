#!/usr/bin/python
import subprocess
import sys

SERVICE_NAME = "music_generator"

subprocess.check_call(["docker", "compose", "run", "--rm", "--build", SERVICE_NAME] + sys.argv[1:])

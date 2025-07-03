#!/bin/bash

echo "eval $(ssh-agent)" > ${HOME}/.bash_profile
git config core.autocrlf false

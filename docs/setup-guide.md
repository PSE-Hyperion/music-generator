# Instructions for setting up the development environment
## Prerequisits
1. Install [Git](https://git-scm.com/downloads)
2. Install [Docker Desktop](https://docs.docker.com/desktop/)
3. Install Python (optional)
## Cloning the repository
### Using SSH authentication (recommended)
1. Create ssh key ([Tutorial for all plattforms](https://www.digitalocean.com/community/tutorials/how-to-create-ssh-keys-with-openssh-on-macos-or-linux))
2. Add public key to github in: Settings -> SSH and GPG keys
3. In a terminal, cd to the preferred directory and type: `git clone git@github.com:PSE-Hyperion/music-generator.git`
### Using HTTP
1. In a terminal, cd to the preferred directory and type: `git clone https://github.com/PSE-Hyperion/music-generator.git`
2. Log in with your github account
## Setting up the docker container
1. Install Docker ([Installation guide for Windows](https://docs.docker.com/desktop/setup/install/windows-install/))
2. In the project repository root, run the script run.py (double click or type `python run.py` in the console)
3. Hope that it works (otherwise write me)
There should pop up a console window with the interactive Python interpreter. GPU support for NVIDIA should be possible, but is not enabled in the default settings.
## VS Code
1. Install the Dev Container extension in VS Code
2. It should recognize the .devcontainer file in the repo and show a message. Otherwise press F1 and type something like 'Dev: open container', then select the .devcontainer file.
VS Code will now run inside the Docker container. All necessary extensions, style checks etc. should get installed automatically.
# Working with the tools
## Docker
### Rebuilding an image
After editing (or pulling changes of) the docker files, you should rebuil the images. `docker compose -f .docker/docker-compose.yml`
You should also rebuild the dev container after doing this.
### Cleanup
You should clean your docker system from time to time. Use `docker system prune -a --volumes` to clean everything up. Keep in mind that your computer has to download everything again after doing this.

## Python/pip
### Installing new packages
1. Type in the dev container (or docker) console `pip install [package]`
The installation will be removed after the next rebuild. Tomake it permanently:
2. Add the package to pyproject.toml in the dependencies with the version number. `[package]>=[version]`
You can see the version with `pip show [package]`

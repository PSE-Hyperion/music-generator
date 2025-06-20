# Project Structure
## Repository root (dirs, files)
- `data` data that will be used/created by the program
- `src` source files of the program
- `docs` documentation of the program and workflows
- `project-assets` assets for the project that are not directly related to the program
- `.devcontainer.json` used by the editor/IDE to run itself in a docker container with all necessary extensions
- `.editorconfig` custom text format standards, used bu the editor/IDE automatically
- `.gitignore` files that shouldn't be tracked by git for a clean repository
- `Dockerfile` used by Docker and Dev Container for building a Docker image
- `LICENSE` formal license specification for the repository
- `README.md` relevant information for all users
- `pdm.lock` fixed version numbers for all used packages, used for building the docker image
- `pyproject.toml` formal specification for all Python projects, stores dependencies
- `run.py` startup script for starting a new Docker Container with Python shell

## Program Structure
![Program Structure Chart](./media/project-structure-chart.png)

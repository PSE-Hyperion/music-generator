# Project Structure
## Repository root (dirs, files)
- `data` data that will be used/created by the program
- `src` source files of the program
- `docs` documentation of the program and workflows
- `scripts` project setup scripts
- `.gitignore` files that shouldn't be tracked by git for a clean repository
- `LICENSE` formal license specification for the repository
- `README.md` relevant information for all users
- `pyproject.toml` formal specification for all Python projects, stores dependencies
- `requirements.txt` contains all dependencies installed with pip

## Program Flow
![Program Flow](./media/program_flow.svg)

## Data Package/Module Structure
![Data Structure](./data_directory_structure.txt)

## Source Package/Module Structure
![Source Structure](./source_directory_structure.txt)
 
## Current dependency structure of packages/modules
Internal: ![Program Dependencies Internal](./media/groove_panda_internal.svg)
External: ![Program Dependencies External](./media/groove_panda.svg)

# Development & Installation

If you want to develop this plugin locally, you can set up a standalone environment or use the NOMAD development distribution.

## Standalone Setup
Clone the project and create a virtual environment (Python 3.10, 3.11, or 3.12):
```sh
git clone https://github.com/FAIRmat-NFDI/nomad-parser-python-workflow-definition.git
cd nomad-parser-python-workflow-definition
python3.11 -m venv .pyenv
. .pyenv/bin/activate
```
Upgrade `pip` and install `uv` for faster installation:
```sh
pip install --upgrade pip
pip install uv
```
Install the package in editable mode with development dependencies:
```sh
uv pip install -e '.[dev]'
```
## Development within NOMAD Distribution
For full integration testing with the NOMAD infrastructure (GUI, North, etc.), we recommend using the dedicated [nomad-distro-dev](https://github.com/FAIRmat-NFDI/nomad-distro-dev) repository. Please refer to that repository for detailed setup instructions.

## Run Tests and Linting
You can run the tests locally:
```sh
python -m pytest -sv tests
```
To run linting and auto-formatting (Ruff):
```sh
ruff format .
ruff check . --fix
```
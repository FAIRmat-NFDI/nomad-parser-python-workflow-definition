# NOMAD Parser for Python Workflow Definition (PWD)

## 1. Introduction & Scope
The **nomad-parser-python-workflow-definition** is a NOMAD plugin designed to bridge the gap between **NOMAD** and the [**Python Workflow Definition (PWD)**](https://github.com/pythonworkflow/python-workflow-definition) standard.

**Scope:** The plugin enables NOMAD to ingest, parse, and visualize scientific workflows defined in the PWD JSON format. This allows workflows orchestrated by external tools (such as [`pyiron`](https://github.com/pyiron/pyiron), [`AiiDA`](https://github.com/aiidateam/aiida-core), or [`jobflow`](https://github.com/materialsproject/jobflow)) to be archived and explored within the NOMAD interface.

## 2. Connecting to PWD
The plugin connects to PWD by interpreting the standardized JSON graph structure:
* **Input:** It accepts `workflow.json` files defined by the PWD standard. This standard structure has been slightly adjusted to add new fields (`output` and `working_directory`) to the nodes in the [forked PWD repo](https://github.com/FAIRmat-NFDI/python-workflow-definition) that is used here.
* **Mapping:** It maps PWD nodes and edges to NOMAD's internal `Workflow` and `Task` schema.
* **Validation:** It utilizes the upstream [`python-workflow-definition`](https://github.com/FAIRmat-NFDI/python-workflow-definition) library to ensure schema compatibility.

## 3. Key Features & Functionality

### A. Smart Graph Simplification
Raw PWD graphs often contain many "utility" nodes (e.g., parameter setters, getters) that clutter the visualization. This plugin implements a **Smart Graph Simplification** strategy using `networkx`:
* **Scientific Tasks:** Nodes that generate files (possess a `working_directory`) or act as topological "Hubs" (high connectivity) remain at the top level.
* **Utility Functions:** Low-level helper nodes are automatically grouped into a nested sub-workflow named "Utility Functions," ensuring the user sees a clean, high-level view of the scientific process.

### B. Data Linking & Path Resolution
The plugin extends the schema to make the graph interactive:
* **Schema Extension:** Adds `working_directory` and `output` fields to capture execution context.
* **Active Linking:** It resolves absolute paths found in the JSON (e.g., `/home/cluster/...`) relative to the NOMAD upload directory. If a simulation file (e.g., `.out`, `.xml`) is found, the graph node becomes a clickable **TaskReference** linking directly to the data entry.

### C. Robust Parsing
The parser uses a robust detection mechanism (scanning for `"nodes": [...]`) to identify PWD files. This ensures reliable ingestion even during streaming uploads where standard JSON parsing might fail on partial buffers.

## 4. Development & Installation

If you want to develop this plugin locally, you can set up a standalone environment or use the NOMAD development distribution.

### Standalone Setup
Clone the project and create a virtual environment (Python 3.10, 3.11, or 3.12):
```sh
git clone [https://github.com/FAIRmat-NFDI/nomad-parser-python-workflow-definition.git](https://github.com/FAIRmat-NFDI/nomad-parser-python-workflow-definition.git)
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
### Development within NOMAD Distribution
For full integration testing with the NOMAD infrastructure (GUI, North, etc.), we recommend using the dedicated [nomad-distro-dev](https://github.com/FAIRmat-NFDI/nomad-distro-dev) repository. Please refer to that repository for detailed setup instructions.

### Run Tests and Linting
You can run the tests locally:
```sh
python -m pytest -sv tests
```
To run linting and auto-formatting (Ruff):
```sh
ruff format .
ruff check . --fix
```

## 5. Demo & Usage

To demonstrate the plugin functionality:

### Prerequisites
A valid upload must contain the following "Companion Files":
* `workflow.json` (The graph structure)
* `workflow.py` (The Python script definition)
* `environment.yaml` (Dependency definition)

### Verification Steps
1.  **Start the GUI:** Run your local NOMAD instance and navigate to the UI.
2.  **Upload:** Create a new upload and drop the example files (e.g., files in `tests/data/data_export/`).
3.  **Visual Verification:**
    * **Graph View:** Open the Workflow entry. You should see a simplified graph where utility nodes are hidden.
    * **Data Links:** To test this feature, we can use the mock simulation directories zipped in the above test data directory. Click on a compute node. If the `working_directory` contains output files, NOMAD will navigate you to that specific file entry.

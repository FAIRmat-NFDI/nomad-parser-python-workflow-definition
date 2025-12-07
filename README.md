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

## 4. Installation

This plugin is designed to be installed within a [NOMAD distribution](https://github.com/FAIRmat-NFDI/nomad-distro-dev).

1.  **Add to Workspace:**
    Ensure the plugin is added to your `packages/` directory and registered in `pyproject.toml` as a workspace source:
    ```toml
    [tool.uv.sources]
    nomad-parser-pwd = { workspace = true }
    ```
2.  **Install Dependencies:**
    Run the setup command to install the plugin and its dependencies:
    ```bash
    uv sync
    ```

## 5. Demo & Usage

To demonstrate the plugin functionality:

### Prerequisites
A valid upload must contain the following "Companion Files":
* `workflow.json` (The graph structure)
* `workflow.py` (The Python script definition)
* `environment.yaml` (Dependency definition)

### Verification Steps
1.  **Start the GUI:** Run `uv run poe gui start` and navigate to `http://localhost:3000`.
2.  **Upload:** Create a new upload and drop the example files (e.g., files in `tests/data/data_export/`).
3.  **Visual Verification:**
    * **Graph View:** Open the Workflow entry. You should see a simplified graph where utility nodes are hidden.
    * **Data Links:** Click on a compute node. If the `working_directory` contains output files, NOMAD will navigate you to that specific file entry.

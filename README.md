# NOMAD Parser for Python Workflow Definition (PWD)

## 1. Introduction & Scope
[cite_start]The **nomad-parser-pwd** is a NOMAD plugin designed to bridge the gap between **NOMAD** and the **Python Workflow Definition (PWD)** standard[cite: 2].

[cite_start]**Scope:** The plugin enables NOMAD to ingest, parse, and visualize scientific workflows defined in the PWD JSON format[cite: 20, 39]. [cite_start]This allows workflows orchestrated by external tools (such as `pyiron`, `AiiDA`, or `jobflow`) to be archived and explored within the NOMAD interface[cite: 2].

## 2. Connecting to PWD
The plugin connects to PWD by interpreting the standardized JSON graph structure:
* [cite_start]**Input:** It accepts `workflow.json` files defined by the PWD standard[cite: 39].
* [cite_start]**Mapping:** It maps PWD nodes and edges to NOMAD's internal `Workflow` and `Task` schema[cite: 238].
* [cite_start]**Validation:** It utilizes the upstream `python-workflow-definition` library to ensure schema compatibility[cite: 54, 191].

## 3. Key Features & Functionality

### A. Smart Graph Simplification
[cite_start]Raw PWD graphs often contain many "utility" nodes (e.g., parameter setters, getters) that clutter the visualization[cite: 241]. [cite_start]This plugin implements a **Smart Graph Simplification** strategy using `networkx`[cite: 367]:
* [cite_start]**Scientific Tasks:** Nodes that generate files (possess a `working_directory`) or act as topological "Hubs" (high connectivity) remain at the top level[cite: 369, 370].
* [cite_start]**Utility Functions:** Low-level helper nodes are automatically grouped into a nested sub-workflow named "Utility Functions," ensuring the user sees a clean, high-level view of the scientific process[cite: 271, 371].

### B. Data Linking & Path Resolution
The plugin extends the schema to make the graph interactive:
* [cite_start]**Schema Extension:** Adds `working_directory` and `output` fields to capture execution context[cite: 23, 24].
* [cite_start]**Active Linking:** It resolves absolute paths found in the JSON (e.g., `/home/cluster/...`) relative to the NOMAD upload directory[cite: 322, 325]. [cite_start]If a simulation file (e.g., `.out`, `.xml`) is found, the graph node becomes a clickable **TaskReference** linking directly to the data entry[cite: 319, 320].

### C. Robust Parsing
[cite_start]The parser uses a robust regex-based detection mechanism (scanning for `"nodes": [...]`) to identify PWD files[cite: 377]. [cite_start]This ensures reliable ingestion even during streaming uploads where standard JSON parsing might fail on partial buffers[cite: 235].

## 4. Installation

This plugin is designed to be installed within a NOMAD distribution.

1.  **Add to Workspace:**
    Ensure the plugin is added to your `packages/` directory and registered in `pyproject.toml` as a workspace source:
    ```toml
    [tool.uv.sources]
    nomad-parser-pwd = { workspace = true }
    ```
2.  **Install Dependencies:**
    Run the setup command to install the plugin and its dependencies (including `networkx`):
    ```bash
    uv sync
    ```

## 5. Demo & Usage

To demonstrate the plugin functionality:

### Prerequisites
[cite_start]A valid upload must contain the following "Companion Files"[cite: 236]:
* `workflow.json` (The graph structure)
* `workflow.py` (The Python script definition)
* `environment.yaml` (Dependency definition)

### Verification Steps
1.  [cite_start]**Start the GUI:** Run `uv run poe gui start` and navigate to `http://localhost:3000`[cite: 72, 74].
2.  [cite_start]**Upload:** Create a new upload and drop the example files (e.g., `tests/data/arithmetic/workflow.json`)[cite: 75].
3.  **Visual Verification:**
    * **Graph View:** Open the Workflow entry. [cite_start]You should see a simplified graph where utility nodes are hidden[cite: 76].
    * **Data Links:** Click on a compute node. [cite_start]If the `working_directory` contains output files, NOMAD will navigate you to that specific file entry[cite: 26].
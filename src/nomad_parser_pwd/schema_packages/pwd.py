#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
from typing import Any

from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.workflow import Link, Task, Workflow
from nomad.metainfo import JSON, Package, Quantity, Section, SubSection

logger = logging.getLogger(__name__)

m_package = Package(name='python_workflow_definition')


class WorkflowAuthor(ArchiveSection):
    """
    Author information for workflow definition.
    Following h5md pattern for creator tracking.
    """

    m_def = Section(validate=False)

    name = Quantity(
        type=str,
        description='Name of workflow definition author.',
    )

    email = Quantity(
        type=str,
        description='Email of workflow definition author.',
    )

    affiliation = Quantity(
        type=str,
        description='Institution or organization of the author.',
    )


class WorkflowExecutionEnvironment(ArchiveSection):
    """
    Execution environment details for workflow.
    """

    m_def = Section(validate=False)

    python_version = Quantity(
        type=str,
        description='Python interpreter version used.',
    )

    packages = Quantity(
        type=str,
        shape=['*'],
        description='Python packages and versions available in environment.',
    )

    platform = Quantity(
        type=str,
        description='Operating system and platform information.',
    )


class PythonWorkflowDefinitionNode(ArchiveSection):
    """
    Section representing a single node from the original PWD workflow definition.
    Can represent input values, output placeholders, or function specifications.
    """

    m_def = Section(validate=False)

    node_id = Quantity(
        type=int,
        description='Unique identifier of the node in the workflow definition.',
    )

    node_type = Quantity(
        type=str,
        description=('Type of the node: "input", "output", or "function".'),
    )

    name = Quantity(
        type=str,
        description='Name or label of the node.',
    )

    value = Quantity(
        type=JSON,
        description=(
            'Value associated with the node. For input nodes, this contains '
            'the input value. For function nodes, this contains the function '
            'specification. For output nodes, this may be empty or contain '
            'placeholder information.'
        ),
    )


class PythonWorkflowDefinitionMethod(ArchiveSection):
    """
    Method section containing workflow definition metadata and configuration.
    Following established NOMAD workflow patterns.
    """

    m_def = Section(validate=False)

    # Core specification metadata
    version = Quantity(
        type=str,
        description='Python Workflow Definition specification version.',
    )

    # Original workflow structure counts (for statistics)
    original_node_count = Quantity(
        type=int,
        description='Number of nodes in the original workflow definition.',
    )

    original_edge_count = Quantity(
        type=int,
        description='Number of edges in the original workflow definition.',
    )

    # Author/creator information
    author = SubSection(
        sub_section=WorkflowAuthor,
        description='Workflow definition author information.',
    )

    # Execution environment details
    execution_environment = SubSection(
        sub_section=WorkflowExecutionEnvironment,
        description='Runtime environment and platform details.',
    )


class PythonWorkflowDefinitionResults(ArchiveSection):
    """
    Results section containing workflow execution results and derived properties.
    Following established NOMAD workflow patterns - only outputs, no metadata.
    """

    m_def = Section(validate=False)

    execution_time = Quantity(
        type=float,
        unit='second',
        description='Total execution time of the workflow.',
    )

    execution_successful = Quantity(
        type=bool,
        description='Whether the workflow execution completed successfully.',
    )

    final_outputs = Quantity(
        type=JSON,
        description='Final output values from the workflow execution.',
    )

    # Compatibility properties for tests - delegate to parent workflow
    @property
    def n_nodes(self) -> int:
        """Get the number of nodes from the original workflow definition."""
        if (
            hasattr(self, 'm_parent')
            and hasattr(self.m_parent, 'method')
            and self.m_parent.method
            and self.m_parent.method.original_node_count is not None
        ):
            return self.m_parent.method.original_node_count
        return 0

    @property
    def n_edges(self) -> int:
        """Get the number of edges from the original workflow definition."""
        if (
            hasattr(self, 'm_parent')
            and hasattr(self.m_parent, 'method')
            and self.m_parent.method
            and self.m_parent.method.original_edge_count is not None
        ):
            return self.m_parent.method.original_edge_count
        return 0

    @property
    def n_function_nodes(self) -> int:
        """Get the number of function nodes from the parent workflow."""
        if hasattr(self, 'm_parent') and hasattr(self.m_parent, 'n_function_nodes'):
            return self.m_parent.n_function_nodes
        return 0


class PythonWorkflowDefinitionTask(Task):
    """
    Task representing a single node execution in the Python workflow definition.
    Uses the base Task class inputs/outputs for connections.
    """

    node_type = Quantity(
        type=str,
        description='Type of the node: input, output, or function.',
    )

    node_id = Quantity(
        type=int,
        description='Unique identifier of the node in the workflow definition.',
    )

    module_function = Quantity(
        type=str,
        description=(
            'Function specification in module.function format (for function nodes).'
        ),
    )


class PythonWorkflowDefinition(Workflow):
    """
    Schema for Python Workflow Definition workflows.

    This schema represents workflows defined using the Python Workflow Definition
    specification. It extends the NOMAD Workflow base class and follows the
    established pattern of method/results sections for metadata and outputs.

    Design Notes:
    - Statistics properties computed from native NOMAD structures (tasks/inputs/outputs)
    - Task inputs/outputs use base Task class connections, not custom data sections
    - No JSON duplication - structured data is stored directly in schema quantities
    - Original workflow data stored in method section for Pydantic model reconstruction
    """

    method = SubSection(
        sub_section=PythonWorkflowDefinitionMethod.m_def,
        description='Method and metadata section for the Python workflow definition',
    )

    results = SubSection(
        sub_section=PythonWorkflowDefinitionResults.m_def,
        description='Results section for workflow execution outcomes',
    )

    # PWD nodes representing all nodes from the original workflow definition
    pwd_nodes = SubSection(
        sub_section=PythonWorkflowDefinitionNode,
        repeats=True,
        description=(
            'All nodes from the original PWD workflow definition. This includes '
            'input values, function specifications, and output placeholders.'
        ),
    )

    def __init__(self, m_def=None, m_context=None, **kwargs):
        super().__init__(m_def, m_context, **kwargs)

    @property
    def version(self) -> str | None:
        """Get the workflow version from the method section."""
        if self.method and self.method.version:
            return self.method.version
        return None

    @property
    def workflow_tasks(self):
        """Compatibility property for accessing tasks (for tests)."""
        return self.tasks

    # Statistics properties for workflow analysis
    @property
    def n_nodes(self) -> int:
        """Get the number of nodes from native NOMAD structures."""
        # Count workflow inputs, outputs, and function tasks
        return len(self.inputs) + len(self.outputs) + len(self.tasks)

    @property
    def n_edges(self) -> int:
        """Get the number of edges by counting task connections."""
        edge_count = 0
        for task in self.tasks:
            edge_count += len(task.inputs) + len(task.outputs)
        return edge_count

    @property
    def n_function_nodes(self) -> int:
        """Get the number of function nodes."""
        # Function nodes are represented as tasks
        return len(self.tasks)

    def normalize(self, archive, logger):
        """
        Normalize the Python workflow definition.

        This method processes the workflow definition and creates appropriate
        NOMAD workflow structures from the Python workflow definition.
        """
        super().normalize(archive, logger)

        # The workflow is populated by the parser directly from the JSON structure
        # No additional processing needed here since the parser extracts
        # structured data directly into the schema quantities

    def _create_nomad_workflow_structure(self, nodes, edges):
        """
        Create NOMAD workflow structure from PWD data.
        Stores data in pwd_nodes and creates proper connections.
        """
        # Ensure method section exists for version storage
        if not self.method:
            self.method = PythonWorkflowDefinitionMethod()

        # Create PWD node sections for all nodes
        self._create_pwd_nodes(nodes)

        # Create function tasks with connections to PWD node sections
        self._create_function_tasks_with_pwd_nodes(nodes, edges)

    def _create_pwd_nodes(self, nodes):
        """Create PWD node sections for all nodes in the workflow."""
        for node in nodes:
            node_section = PythonWorkflowDefinitionNode()
            node_section.node_id = node.get('id')
            node_section.node_type = node.get('type')
            node_section.name = node.get('name', f'{node.get("type")}_{node.get("id")}')

            # Handle different value types - JSON field expects dict
            node_value = node.get('value')
            if node_value is not None:
                # Wrap all values in a dict for JSON storage
                if isinstance(node_value, str | int | float | bool | list):
                    node_section.value = {'data': node_value}
                elif isinstance(node_value, dict):
                    # If it's already a dict, ensure it's properly structured
                    node_section.value = node_value
                else:
                    # For other types, wrap them
                    node_section.value = {'data': node_value}
            else:
                node_section.value = None

            self.pwd_nodes.append(node_section)

            # Add to workflow-level inputs/outputs based on node type
            if node.get('type') == 'input':
                self.inputs.append(Link(name=node_section.name, section=node_section))
            elif node.get('type') == 'output':
                self.outputs.append(Link(name=node_section.name, section=node_section))

    def _create_function_tasks_with_pwd_nodes(self, nodes, edges):
        """Create tasks for function nodes with connections to PWD nodes."""
        # Create a mapping from node_id to PWD node section for easy lookup
        node_to_pwd_node = {v.node_id: v for v in self.pwd_nodes}

        # 1. Pre-calculate output ports for function-to-function edges
        output_port_values = self._resolve_intermediate_ports(nodes, edges)

        # 2. Create tasks for each function node
        for node in nodes:
            if node.get('type') != 'function':
                continue
            # We pass 'nodes' here so the output helper can look up target nodes
            self._create_single_function_task(
                node, nodes, edges, output_port_values, node_to_pwd_node
            )

    def _resolve_intermediate_ports(self, nodes, edges):
        """Identify edges between functions and create intermediate output ports."""
        output_port_values = {}
        for edge in edges:
            source_id = edge.get('source')
            source_port = edge.get('sourcePort') or 'result'
            target_id = edge.get('target')

            source_node = next((n for n in nodes if n.get('id') == source_id), None)
            target_node = next((n for n in nodes if n.get('id') == target_id), None)

            # Check if this is a function-to-function connection
            is_func_to_func = (
                source_node
                and source_node.get('type') == 'function'
                and target_node
                and target_node.get('type') == 'function'
            )

            if is_func_to_func:
                key = (source_id, source_port)
                if key not in output_port_values:
                    output_value = PythonWorkflowDefinitionNode()
                    # Generate a unique ID
                    new_id = len(self.pwd_nodes) + len(output_port_values) + 1000
                    output_value.node_id = new_id
                    output_value.node_type = 'output_port'
                    output_value.name = f'{source_id}_{source_port}'
                    output_value.value = {
                        'port': source_port,
                        'source_node': source_id,
                    }

                    output_port_values[key] = output_value
                    self.pwd_nodes.append(output_value)
        return output_port_values

    def _create_single_function_task(
        self, node, nodes, edges, output_port_values, node_to_pwd_node
    ):  # noqa: PLR0913
        """Create and configure a single task from a function node."""
        node_id = node.get('id')
        node_name = f'Function {node.get("value", node_id)}'

        task = PythonWorkflowDefinitionTask(
            name=node_name,
            node_type='function',
            node_id=node_id,
            module_function=node.get('value'),
        )

        # Helper to find inputs
        self._add_task_inputs(
            task, node_id, edges, output_port_values, node_to_pwd_node
        )
        # Helper to find outputs (Passing 'nodes' correctly now)
        self._add_task_outputs(
            task, nodes, node_id, edges, output_port_values, node_to_pwd_node
        )

        self.tasks.append(task)

    def _add_task_inputs(
        self, task, node_id, edges, output_port_values, node_to_pwd_node
    ):  # noqa: PLR0913
        """Find edges targeting this node and add them as inputs."""
        for edge in edges:
            if edge.get('target') != node_id:
                continue

            source_id = edge.get('source')
            source_port = edge.get('sourcePort') or 'result'
            target_port = edge.get('targetPort', f'input_{source_id}')

            # Find the appropriate source section
            source_section = output_port_values.get((source_id, source_port))
            if not source_section:
                source_section = node_to_pwd_node.get(source_id)

            if source_section:
                task.inputs.append(Link(name=target_port, section=source_section))

    def _add_task_outputs( # noqa: PLR0913
        self, task, nodes, node_id, edges, output_port_values, node_to_pwd_node
    ):  
        """Find edges originating from this node and add them as outputs."""
        for edge in edges:
            if edge.get('source') != node_id:
                continue

            source_port = edge.get('sourcePort') or 'result'
            target_id = edge.get('target')
            target_node = next((n for n in nodes if n.get('id') == target_id), None)

            # Find the appropriate output section
            output_section = output_port_values.get((node_id, source_port))

            # Fallback logic if not in intermediate values
            if not output_section:
                if target_node and target_node.get('type') == 'output':
                    # Function-to-output: point to the actual output node
                    output_section = node_to_pwd_node.get(target_id)
                else:
                    # Fallback to target
                    output_section = node_to_pwd_node.get(target_id)

            if output_section:
                task.outputs.append(Link(name=source_port, section=output_section))

    def _create_node_sections(self, nodes):
        """Create minimal sections for nodes that need Link references."""
        node_sections = {}

        for node in nodes:
            node_id = node.get('id')
            node_type = node.get('type')
            node_name = node.get('name', f'{node_type}_{node_id}')

            # Create a proper task section that can be referenced
            if node_type == 'input':
                # For inputs, create a simple task with the input value
                section = PythonWorkflowDefinitionTask()
                section.name = node_name
                section.node_type = 'input'
                section.node_id = node_id
                node_sections[node_id] = section
                self.inputs.append(Link(name=node_name, section=section))

            elif node_type == 'output':
                # For outputs, create a simple task to represent the output
                section = PythonWorkflowDefinitionTask()
                section.name = node_name
                section.node_type = 'output'
                section.node_id = node_id
                node_sections[node_id] = section
                self.outputs.append(Link(name=node_name, section=section))

        return node_sections

    def _create_function_tasks(
        self, nodes, edges, node_sections, connection_id_counter
    ):
        """Create tasks for function nodes with proper connections."""
        current_connection_id = connection_id_counter

        # Create a mapping from (source_node, source_port) to shared section
        # This ensures outputs and inputs reference the same section
        connection_sections = {}

        for node in nodes:
            if node.get('type') != 'function':
                continue

            node_id = node.get('id')
            node_name = f'Function {node.get("value", node_id)}'

            # Create task
            task = PythonWorkflowDefinitionTask(
                name=node_name,
                node_type='function',
                node_id=node_id,
                module_function=node.get('value'),
            )

            # Add input connections based on edges
            for edge in edges:
                if edge.get('target') == node_id:
                    source_id = edge.get('source')
                    source_port = edge.get('sourcePort')
                    if source_port is None:
                        source_port = 'result'
                    target_port = edge.get('targetPort', f'input_{source_id}')

                    # Get or create shared section for this connection
                    connection_key = (source_id, source_port)
                    if connection_key not in connection_sections:
                        # Check if source is an input/output node
                        source_section = node_sections.get(source_id)
                        if source_section:
                            connection_sections[connection_key] = source_section
                        else:
                            # Create shared section for function-to-function connection
                            connection_section = PythonWorkflowDefinitionTask()
                            connection_section.name = (
                                f'connection_{source_id}_{source_port}'
                            )
                            connection_section.node_type = 'connection'
                            connection_section.node_id = current_connection_id
                            current_connection_id += 1
                            connection_sections[connection_key] = connection_section

                    # Add input that references the shared section
                    shared_section = connection_sections[connection_key]
                    task.inputs.append(Link(name=target_port, section=shared_section))

            # Add output connections based on edges
            for edge in edges:
                if edge.get('source') == node_id:
                    source_port = edge.get('sourcePort')
                    if source_port is None:
                        source_port = 'result'

                    # Get or create shared section for this output
                    connection_key = (node_id, source_port)
                    if connection_key not in connection_sections:
                        connection_section = PythonWorkflowDefinitionTask()
                        connection_section.name = f'connection_{node_id}_{source_port}'
                        connection_section.node_type = 'connection'
                        connection_section.node_id = current_connection_id
                        current_connection_id += 1
                        connection_sections[connection_key] = connection_section

                    # Add output that references the shared section
                    shared_section = connection_sections[connection_key]
                    task.outputs.append(Link(name=source_port, section=shared_section))

            self.tasks.append(task)

    def load_from_pydantic_model(self, pwd_workflow):
        """
        Load workflow data from a Pydantic model instance or dict.

        Args:
            pwd_workflow: Instance of PythonWorkflowDefinitionWorkflow or dict
        """
        # Initialize method section if needed
        if not self.method:
            self.method = PythonWorkflowDefinitionMethod()

        # Initialize results section if needed
        if not self.results:
            self.results = PythonWorkflowDefinitionResults()

        # Handle both dict and Pydantic model objects
        if isinstance(pwd_workflow, dict):
            # Extract metadata into method section
            self.method.version = pwd_workflow.get('version')

            # Store original counts for statistics
            nodes = pwd_workflow.get('nodes', [])
            edges = pwd_workflow.get('edges', [])
            self.method.original_node_count = len(nodes)
            self.method.original_edge_count = len(edges)
        else:
            # Extract metadata into method section
            self.method.version = pwd_workflow.version

            # Process nodes and edges to create NOMAD structures
            nodes = [node.model_dump() for node in pwd_workflow.nodes]
            edges = [edge.model_dump() for edge in pwd_workflow.edges]
            self.method.original_node_count = len(nodes)
            self.method.original_edge_count = len(edges)

        # Create workflow tasks and NOMAD workflow structure
        self._create_nomad_workflow_structure(nodes, edges)

    def get_pydantic_model(self) -> dict[str, Any] | None:
        """
        Reconstruct PWD from NOMAD workflow structures.

        Returns:
            Dictionary representation that matches PythonWorkflowDefinitionWorkflow
        """
        if not self.method or not self.method.version:
            return None

        try:
            nodes = []
            edges = []

            # Reconstruct input nodes from workflow.inputs
            for i, workflow_input in enumerate(self.inputs):
                nodes.append(
                    {
                        'id': i + 1,  # Assign sequential IDs
                        'type': 'input',
                        'name': workflow_input.name,
                        'value': None,
                    }
                )

            # Reconstruct output nodes from workflow.outputs
            max_input_id = len(self.inputs)
            for i, workflow_output in enumerate(self.outputs):
                nodes.append(
                    {
                        'id': max_input_id + i + 1,
                        'type': 'output',
                        'name': workflow_output.name,
                        'value': None,
                    }
                )

            # Reconstruct function nodes from tasks
            max_io_id = len(self.inputs) + len(self.outputs)
            for i, task in enumerate(self.tasks):
                nodes.append(
                    {
                        'id': max_io_id + i + 1,
                        'type': 'function',
                        'name': task.name,
                        'value': task.module_function,
                    }
                )

            # Reconstruct edges from task connections
            # This would require matching Link sections between tasks
            # For now, return empty edges as proof of concept

            return {'version': self.method.version, 'nodes': nodes, 'edges': edges}

        except Exception as e:
            logger.error(f'Failed to reconstruct PWD from NOMAD structures: {e}')
            return None


m_package.__init_metainfo__()

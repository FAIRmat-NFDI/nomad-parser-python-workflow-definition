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
from python_workflow_definition.models import (
    INTERNAL_DEFAULT_HANDLE,
)

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


class PythonWorkflowDefinitionValue(ArchiveSection):
    """
    Section representing a workflow value (input, output, or intermediate).

    This section stores actual values from the workflow definition and provides
    a section reference for workflow connections.
    """

    m_def = Section(validate=False)

    name = Quantity(
        type=str,
        description='Name/label of the value.',
    )

    value = Quantity(
        type=Any,
        description='The actual value stored.',
    )

    node_id = Quantity(
        type=int,
        description='ID of the corresponding node in the workflow definition.',
    )

    node_type = Quantity(
        type=str,
        description='Type of the node: input, output, or function.',
    )

    port_name = Quantity(
        type=str,
        description='Port name for function outputs (e.g., sourcePort in edges).',
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

    # Author/creator information
    author = SubSection(
        sub_section=WorkflowAuthor,
        description='Workflow definition author information.',
    )

    # Execution environment details
    python_environment = Quantity(
        type=str,
        description='Python environment or interpreter used for execution.',
    )

    execution_environment = SubSection(
        sub_section=WorkflowExecutionEnvironment,
        description='Runtime environment and platform details.',
    )

    # Original workflow definition data for reconstruction
    workflow_values = SubSection(
        sub_section=PythonWorkflowDefinitionValue.m_def,
        repeats=True,
        description=(
            'Workflow values representing nodes and their data from original definition'
        ),
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
        """Get the number of nodes from the parent workflow."""
        if hasattr(self, 'm_parent') and hasattr(self.m_parent, 'n_nodes'):
            return self.m_parent.n_nodes
        return 0

    @property 
    def n_edges(self) -> int:
        """Get the number of edges from the parent workflow."""
        if hasattr(self, 'm_parent') and hasattr(self.m_parent, 'n_edges'):
            return self.m_parent.n_edges
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
        """Get the number of edges from stored workflow values."""
        # Count connection sections from method.workflow_values
        if self.method and self.method.workflow_values:
            connections = [
                v for v in self.method.workflow_values if v.node_type == 'connection'
            ]
            return len(connections)
        return 0

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

    def _create_nomad_workflow_structure(self, nodes, edges, logger):
        """
        Create NOMAD workflow structure (tasks, inputs, outputs) from the
        Python workflow definition.

        This method creates proper Link connections between tasks based on edges,
        ensuring each edge creates exactly matching input/output connections
        that reference the same section for NOMAD graph visualization.
        """
        # Initialize method.workflow_values if needed
        if not self.method.workflow_values:
            self.method.workflow_values = []

        # Step 1: Create value sections for all nodes
        node_id_to_value_section = {}

        for node in nodes:
            node_id = node.get('id')
            node_type = node.get('type')

            # Create a value section for each node
            value_section = PythonWorkflowDefinitionValue(
                name=node.get('name', f'Node {node_id}'),
                value=node.get('value'),
                node_id=node_id,
                node_type=node_type,
            )

            # Store the section in method
            node_id_to_value_section[node_id] = value_section
            self.method.workflow_values.append(value_section)

            # Add to workflow-level inputs/outputs based on node type
            if node_type == 'input':
                self.inputs.append(
                    Link(
                        name=node.get('name', f'Input {node_id}'), 
                        section=value_section
                    )
                )
            elif node_type == 'output':
                self.outputs.append(
                    Link(
                        name=node.get('name', f'Output {node_id}'),
                        section=value_section,
                    )
                )

        # Step 2: Create connection value sections for edges
        # This ensures each edge has exactly one section that both
        # source output and target input will reference
        edge_to_connection_section = {}
        
        for i, edge in enumerate(edges):
            source_id = edge.get('source')
            target_id = edge.get('target')
            source_port = edge.get('sourcePort')
            target_port = edge.get('targetPort', f'input_{source_id}')
            
            # Handle None and INTERNAL_DEFAULT_HANDLE the same way
            if source_port is None or source_port == INTERNAL_DEFAULT_HANDLE:
                source_port = 'result'
            
            # Create a unique value section for this connection
            connection_name = (
                f'Connection {source_id}:{source_port} -> {target_id}:{target_port}'
            )
            connection_section = PythonWorkflowDefinitionValue(
                name=connection_name,
                value=None,  # Will be populated during execution
                node_id=source_id,  # Track which node produces this value
                node_type='connection',
                port_name=source_port,
            )
            
            self.method.workflow_values.append(connection_section)
            edge_to_connection_section[i] = connection_section

        # Step 3: Create tasks for function nodes and establish connections
        for node in nodes:
            if node.get('type') == 'function':
                node_id = node.get('id')

                # Create task
                task = PythonWorkflowDefinitionTask(
                    name=f"Function {node.get('value', node_id)}",
                    node_type='function',
                    node_id=node_id,
                    module_function=node.get('value'),
                )

                # Step 4: Add connections using edge-specific sections
                self._add_edge_based_connections(
                    task, node_id, edges, edge_to_connection_section
                )

                self.tasks.append(task)

        # Log completion
        if logger:
            logger.info(f'Created {len(self.tasks)} tasks')

    def _add_edge_based_connections(
        self, task, node_id, edges, edge_to_connection_section
    ):
        """
        Add input and output connections based on edges using shared sections.
        
        This ensures each edge creates exactly one input and one output connection
        that reference the same section for proper NOMAD graph visualization.
        """
        # Add input connections: edges where this node is the target
        for i, edge in enumerate(edges):
            if edge.get('target') == node_id:
                source_id = edge.get('source')
                target_port = edge.get('targetPort', f'input_{source_id}')
                
                # Find the source node type to determine connection strategy
                source_node_value = next(
                    (v for v in self.method.workflow_values if v.node_id == source_id 
                     and v.node_type != 'connection'), 
                    None
                )
                
                if source_node_value and source_node_value.node_type == 'input':
                    # Input node case: connect directly to node value section
                    task.inputs.append(
                        Link(name=target_port, section=source_node_value)
                    )
                else:
                    # Function node case: use the connection section
                    connection_section = edge_to_connection_section[i]
                    task.inputs.append(
                        Link(name=target_port, section=connection_section)
                    )

        # Add output connections: edges where this node is the source
        for i, edge in enumerate(edges):
            if edge.get('source') == node_id:
                source_port = edge.get('sourcePort')
                # Handle None and INTERNAL_DEFAULT_HANDLE the same way
                if source_port is None or source_port == INTERNAL_DEFAULT_HANDLE:
                    source_port = 'result'
                
                # Use the connection section for this edge
                connection_section = edge_to_connection_section[i]
                task.outputs.append(
                    Link(name=source_port, section=connection_section)
                )

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
            
            # Process nodes and edges to create NOMAD structures
            nodes = pwd_workflow.get('nodes', [])
            edges = pwd_workflow.get('edges', [])
        else:
            # Extract metadata into method section  
            self.method.version = pwd_workflow.version
            
            # Process nodes and edges to create NOMAD structures
            nodes = [node.model_dump() for node in pwd_workflow.nodes]
            edges = [edge.model_dump() for edge in pwd_workflow.edges]
        
        # Create workflow tasks and NOMAD workflow structure
        self._create_nomad_workflow_structure(nodes, edges, logger)

    def get_pydantic_model(self) -> dict[str, Any] | None:
        """
        Get the workflow as a Pydantic model dict.

        Returns:
            Dictionary representation that matches PythonWorkflowDefinitionWorkflow
        """
        if (not self.method or not self.method.version or 
            not self.method.workflow_values):
            return None
            
        try:
            # Reconstruct the workflow dict from stored structured data
            nodes = []
            edges = []
            
            # Reconstruct nodes from method.workflow_values 
            for value in self.method.workflow_values:
                if value.node_type in ['input', 'output', 'function']:
                    node = {
                        'id': value.node_id,
                        'type': value.node_type,
                    }
                    if value.name:
                        node['name'] = value.name
                    if value.value is not None:
                        node['value'] = value.value
                    nodes.append(node)
            
            # TODO: Reconstruct edges from task connections
            # This would require analyzing the Link connections between tasks
            
            return {
                'version': self.method.version,
                'nodes': nodes,
                'edges': edges
            }
            
        except Exception as e:
            logger.error(f'Failed to reconstruct Pydantic model: {e}')
        return None


m_package.__init_metainfo__()

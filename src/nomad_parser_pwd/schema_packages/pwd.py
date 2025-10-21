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

import json
import logging
from typing import Any

from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.workflow import Link, Task, Workflow
from nomad.metainfo import JSON, Package, Quantity, Section, SubSection
from python_workflow_definition.models import (
    INTERNAL_DEFAULT_HANDLE,
    PythonWorkflowDefinitionWorkflow,
)

logger = logging.getLogger(__name__)

m_package = Package(name='python_workflow_definition')


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


class PythonWorkflowDefinitionInputData(ArchiveSection):
    """
    Section representing input data for a Python workflow definition.
    """

    m_def = Section(validate=False)

    name = Quantity(
        type=str,
        description='Name of the input parameter.',
    )

    value = Quantity(
        type=Any,
        description='Value of the input parameter.',
    )

    node_id = Quantity(
        type=int,
        description='ID of the corresponding input node in the workflow definition.',
    )


class PythonWorkflowDefinitionOutputData(ArchiveSection):
    """
    Section representing output data for a Python workflow definition.
    """

    m_def = Section(validate=False)

    name = Quantity(
        type=str,
        description='Name of the output parameter.',
    )

    value = Quantity(
        type=Any,
        description='Value of the output parameter (if available).',
    )

    node_id = Quantity(
        type=int,
        description='ID of the corresponding output node in the workflow definition.',
    )


class PythonWorkflowDefinitionFunctionData(ArchiveSection):
    """
    Section representing a function execution in a Python workflow definition.
    """

    m_def = Section(validate=False)

    module_function = Quantity(
        type=str,
        description='Function specification in module.function format.',
    )

    node_id = Quantity(
        type=int,
        description='ID of the corresponding function node in the workflow definition.',
    )

    inputs = Quantity(
        type=JSON,
        description='Input parameters for this function.',
    )

    outputs = Quantity(
        type=JSON,
        description='Output values from this function execution.',
    )

    execution_status = Quantity(
        type=str,
        description='Execution status: pending, running, completed, failed.',
    )

    error_message = Quantity(
        type=str,
        description='Error message if execution failed.',
    )


class PythonWorkflowDefinitionMethod(ArchiveSection):
    """
    Method section containing the workflow definition and configuration.
    """

    m_def = Section(validate=False)

    version = Quantity(
        type=str,
        description='Version of the Python workflow definition format.',
    )

    workflow_definition = Quantity(
        type=str,
        description='JSON representation of the workflow definition.',
    )

    python_environment = Quantity(
        type=str,
        description='Python environment or interpreter used for execution.',
    )

    def normalize(self, archive, logger):
        """Normalize the method section."""
        super().normalize(archive, logger)

        # Validate workflow definition JSON if present
        if self.workflow_definition:
            try:
                json.loads(self.workflow_definition)
            except json.JSONDecodeError as e:
                logger.warning(f'Invalid workflow definition JSON: {e}')


class PythonWorkflowDefinitionResults(ArchiveSection):
    """
    Results section containing execution results and metadata.
    """

    m_def = Section(validate=False)

    execution_time = Quantity(
        type=float,
        unit='second',
        description='Total execution time of the workflow.',
    )

    n_nodes = Quantity(
        type=int,
        description='Total number of nodes in the workflow.',
    )

    n_edges = Quantity(
        type=int,
        description='Total number of edges in the workflow.',
    )

    n_function_nodes = Quantity(
        type=int,
        description='Number of function nodes in the workflow.',
    )

    execution_successful = Quantity(
        type=bool,
        description='Whether the workflow execution completed successfully.',
    )

    final_outputs = Quantity(
        type=JSON,
        description='Final output values from the workflow execution.',
    )


class PythonWorkflowDefinitionTask(Task):
    """
    Task representing a single node execution in the Python workflow definition.
    """

    node_type = Quantity(
        type=str,
        description='Type of the node: input, output, or function.',
    )

    node_id = Quantity(
        type=int,
        description='Unique identifier of the node in the workflow definition.',
    )

    function_data = SubSection(
        sub_section=PythonWorkflowDefinitionFunctionData,
        description='Function execution data (for function nodes).',
    )

    input_data = SubSection(
        sub_section=PythonWorkflowDefinitionInputData,
        description='Input data (for input nodes).',
    )

    output_data = SubSection(
        sub_section=PythonWorkflowDefinitionOutputData,
        description='Output data (for output nodes).',
    )


class PythonWorkflowDefinition(Workflow):
    """
    Main workflow class for Python workflow definitions.

    This class integrates the Python workflow definition models with NOMAD's
    workflow framework, providing storage and execution tracking capabilities.
    """

    method = SubSection(
        sub_section=PythonWorkflowDefinitionMethod,
        description='Method section containing workflow definition and configuration.',
    )

    results = SubSection(
        sub_section=PythonWorkflowDefinitionResults,
        description='Results section containing execution results.',
    )

    workflow_tasks = SubSection(
        sub_section=PythonWorkflowDefinitionTask,
        repeats=True,
        description='Tasks representing individual node executions.',
    )

    # Store all workflow values (inputs, outputs, intermediate results)
    workflow_values = SubSection(
        sub_section=PythonWorkflowDefinitionValue,
        repeats=True,
        description='All values in the workflow (inputs, outputs, intermediate).',
    )

    # Raw workflow data for easy access
    raw_workflow_definition = Quantity(
        type=str,
        description='Raw JSON workflow definition for direct access.',
    )

    def normalize(self, archive, logger):
        """
        Normalize the Python workflow definition.

        This method processes the workflow definition and creates appropriate
        NOMAD workflow structures from the Python workflow definition.
        """
        super().normalize(archive, logger)

        if (
            not self.raw_workflow_definition
            and self.method
            and self.method.workflow_definition
        ):
            self.raw_workflow_definition = self.method.workflow_definition

        if self.raw_workflow_definition:
            try:
                self._process_workflow_definition(logger)
            except Exception as e:
                logger.error(f'Error processing workflow definition: {e}')

    def _process_workflow_definition(self, logger):
        """
        Process the raw workflow definition and create NOMAD structures.
        """
        try:
            # Parse the workflow definition using the Pydantic model
            pwd_workflow = PythonWorkflowDefinitionWorkflow.load_json_str(
                self.raw_workflow_definition
            )
        except Exception as e:
            logger.error(f'Failed to parse workflow definition: {e}')
            return

        # Update method section
        if not self.method:
            self.method = PythonWorkflowDefinitionMethod()

        if not self.method.version:
            self.method.version = pwd_workflow.get('version', 'unknown')

        if not self.method.workflow_definition:
            self.method.workflow_definition = self.raw_workflow_definition

        # Update results section
        if not self.results:
            self.results = PythonWorkflowDefinitionResults()

        # Count nodes and edges
        nodes = pwd_workflow.get('nodes', [])
        edges = pwd_workflow.get('edges', [])

        self.results.n_nodes = len(nodes)
        self.results.n_edges = len(edges)
        self.results.n_function_nodes = len(
            [n for n in nodes if n.get('type') == 'function']
        )

        # Create workflow tasks and NOMAD workflow structure
        self._create_nomad_workflow_structure(nodes, edges, logger)

    def _create_nomad_workflow_structure(self, nodes, edges, logger):
        """
        Create NOMAD workflow structure (tasks, inputs, outputs) from the
        Python workflow definition.
        
        This method creates proper Link connections between tasks based on edges,
        storing values in dedicated sections that can be referenced.
        """
        # Step 1: Create value sections for all nodes
        node_id_to_value_section = {}
        
        for node in nodes:
            node_id = node.get('id')
            node_type = node.get('type')
            
            # Create a value section for each node
            value_section = PythonWorkflowDefinitionValue(
                name=node.get('name', f'Node {node_id}'),
                value=node.get('value'),  # This will be properly handled by JSON type
                node_id=node_id,
                node_type=node_type,
            )
            
            # Store the section
            node_id_to_value_section[node_id] = value_section
            self.workflow_values.append(value_section)
            
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
                        section=value_section
                    )
                )

        # Step 2: Create tasks for function nodes and establish connections
        for node in nodes:
            if node.get('type') == 'function':
                node_id = node.get('id')
                
                # Create task
                task = PythonWorkflowDefinitionTask(
                    name=f"Function {node.get('value', node_id)}",
                    node_type='function',
                    node_id=node_id,
                )
                
                # Create function data section
                function_data = PythonWorkflowDefinitionFunctionData(
                    module_function=node.get('value'),
                    node_id=node_id,
                    execution_status='pending',
                )
                task.function_data = function_data
                
                # Step 3: Add input connections based on edges
                self._add_task_input_connections(
                    task, node_id, edges, node_id_to_value_section
                )
                
                # Step 4: Add output connections based on edges  
                self._add_task_output_connections(
                    task, node_id, edges, node_id_to_value_section
                )
                
                self.workflow_tasks.append(task)

        # Update main workflow tasks (only function tasks)
        self.tasks = [
            task for task in self.workflow_tasks if task.node_type == 'function'
        ]

    def _add_task_input_connections(
        self, task, node_id, edges, node_id_to_value_section
    ):
        """
        Add input connections to a task based on workflow edges.
        
        Each edge where this node is the target creates an input connection.
        """
        for edge in edges:
            if edge.get('target') == node_id:
                source_id = edge.get('source')
                if source_id in node_id_to_value_section:
                    source_section = node_id_to_value_section[source_id]
                    port_name = edge.get('targetPort', f'input_{source_id}')
                    
                    # Create input link
                    task.inputs.append(
                        Link(name=port_name, section=source_section)
                    )

    def _add_task_output_connections(
        self, task, node_id, edges, node_id_to_value_section
    ):
        """
        Add output connections from a task based on workflow edges.
        
        Each edge where this node is the source creates an output connection.
        For function nodes, we may need to create additional value sections
        for different output ports.
        """
        # Group edges by sourcePort to handle multiple outputs
        output_ports = {}
        for edge in edges:
            if edge.get('source') == node_id:
                source_port = edge.get('sourcePort', INTERNAL_DEFAULT_HANDLE)
                if source_port == INTERNAL_DEFAULT_HANDLE:
                    source_port = 'result'
                
                if source_port not in output_ports:
                    output_ports[source_port] = []
                output_ports[source_port].append(edge)
        
        # Create output connections for each port
        for port_name, port_edges in output_ports.items():
            # Create a value section for this output port if it doesn't exist
            output_value_section = PythonWorkflowDefinitionValue(
                name=f'{task.name} - {port_name}',
                node_id=node_id,
                node_type='function',
                port_name=port_name,
            )
            
            # Add to workflow values
            self.workflow_values.append(output_value_section)
            
            # Create output link from task to this value section
            task.outputs.append(
                Link(name=port_name, section=output_value_section)
            )

    def load_from_pydantic_model(self, pwd_workflow: PythonWorkflowDefinitionWorkflow):
        """
        Load workflow data from a Pydantic model instance.

        Args:
            pwd_workflow: Instance of PythonWorkflowDefinitionWorkflow
        """
        self.raw_workflow_definition = pwd_workflow.dump_json()

        # Trigger normalization to process the workflow
        if hasattr(self, 'm_parent') and hasattr(self.m_parent, 'logger'):
            self._process_workflow_definition(self.m_parent.logger)

    def get_pydantic_model(self) -> dict[str, Any] | None:
        """
        Get the workflow as a Pydantic model dict.

        Returns:
            Dictionary representation of the PythonWorkflowDefinitionWorkflow
        """
        if self.raw_workflow_definition:
            try:
                return PythonWorkflowDefinitionWorkflow.load_json_str(
                    self.raw_workflow_definition
                )
            except Exception as e:
                logger.error(f'Failed to create Pydantic model: {e}')
        return None


m_package.__init_metainfo__()

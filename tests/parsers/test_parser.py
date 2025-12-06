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
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nomad.datamodel import EntryArchive
from nomad.datamodel.metainfo.workflow import TaskReference

from nomad_parser_pwd.parsers.parser import PythonWorkflowDefinitionParser
from nomad_parser_pwd.schema_packages.pwd import (
    PythonWorkflowDefinition,
    PythonWorkflowDefinitionTask,
)

# Expected counts for test examples
ARITHMETIC_NODES = 6
ARITHMETIC_EDGES = 6
ARITHMETIC_FUNCTIONS = 3
ARITHMETIC_INPUTS = 2
ARITHMETIC_OUTPUTS = 1

NFDI_NODES = 9
NFDI_EDGES = 17
NFDI_FUNCTIONS = 6

QE_NODES = 33
QE_MIN_EDGES = 20
QE_MIN_FUNCTIONS = 10


@pytest.fixture(scope='module')
def parser():
    """Fixture providing the Python Workflow Definition parser."""
    return PythonWorkflowDefinitionParser()


@pytest.fixture
def test_data_path():
    """Fixture providing the path to test data."""
    return Path(__file__).parent.parent / 'data'


def get_all_tasks(workflow):
    """
    Helper to return a flat list of all tasks, including those inside
    the 'Utility Functions' sub-workflow.
    """
    all_tasks = []
    for task in workflow.tasks:
        if getattr(task, 'name', '') == 'Utility Functions':
            all_tasks.extend(task.tasks)
        else:
            all_tasks.append(task)
    return all_tasks


def test_parser_instantiation(parser):
    """Test that the parser can be instantiated."""
    assert parser is not None
    assert parser.name == 'parsers/python_workflow_definition'
    assert parser.code_name == 'Python Workflow Definition'


def test_is_mainfile_valid_workflow_json(parser, test_data_path):
    """Test is_mainfile with valid workflow.json files."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'

    with open(workflow_json_path, 'rb') as f:
        buffer = f.read()
    with open(workflow_json_path, encoding='utf-8') as f:
        decoded_buffer = f.read()

    result = parser.is_mainfile(
        str(workflow_json_path), 'application/json', buffer, decoded_buffer
    )

    assert result is True


def test_is_mainfile_non_workflow_json(parser):
    """Test is_mainfile rejects non-workflow JSON files."""
    # Test with a non-workflow.json filename
    result = parser.is_mainfile(
        'random_file.json',
        'application/json',
        b'{"version": "1.0"}',
        '{"version": "1.0"}',
    )

    assert result is False


def test_is_mainfile_invalid_json_structure(parser):
    """Test is_mainfile rejects invalid JSON structure."""
    result = parser.is_mainfile(
        'workflow.json',
        'application/json',
        b'{"invalid": "structure"}',
        '{"invalid": "structure"}',
    )

    assert result is False


def test_parse_arithmetic_example(parser, test_data_path):
    """Test parsing the arithmetic workflow example."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    parser.parse(str(workflow_json_path), archive, None)

    # Test workflow creation
    assert archive.workflow2 is not None
    assert hasattr(archive.workflow2, 'name')
    name_lower = archive.workflow2.name.lower()
    assert 'arithmetic' in name_lower or 'python workflow definition' in name_lower

    # Test version at top level (as per specification)
    assert archive.workflow2.version == '0.1.0'

    # Test method section
    assert archive.workflow2.method is not None

    # Test results section
    assert archive.workflow2.results is not None
    assert archive.workflow2.results.n_nodes == ARITHMETIC_NODES
    assert archive.workflow2.results.n_edges == ARITHMETIC_EDGES
    assert archive.workflow2.results.n_function_nodes == ARITHMETIC_FUNCTIONS


def test_parse_nfdi_example(parser, test_data_path):
    """Test parsing the NFDI workflow example."""
    workflow_json_path = test_data_path / 'nfdi' / 'workflow.json'
    archive = EntryArchive()

    parser.parse(str(workflow_json_path), archive, None)

    # Test workflow creation
    assert archive.workflow2 is not None

    # Test results section
    assert archive.workflow2.results is not None
    assert archive.workflow2.results.n_nodes == NFDI_NODES
    assert archive.workflow2.results.n_edges == NFDI_EDGES
    assert archive.workflow2.results.n_function_nodes == NFDI_FUNCTIONS


def test_parse_quantum_espresso_example(parser, test_data_path):
    """Test parsing the Quantum ESPRESSO workflow example."""
    workflow_json_path = test_data_path / 'quantum_espresso' / 'workflow.json'
    archive = EntryArchive()

    parser.parse(str(workflow_json_path), archive, None)

    # Test workflow creation
    assert archive.workflow2 is not None

    # Test results section - QE example has many nodes
    assert archive.workflow2.results is not None
    assert archive.workflow2.results.n_nodes == QE_NODES
    assert archive.workflow2.results.n_edges > QE_MIN_EDGES
    assert archive.workflow2.results.n_function_nodes > QE_MIN_FUNCTIONS


def test_parse_missing_companion_files(parser, test_data_path, tmp_path):
    """Test parsing fails when companion files are missing."""
    # Create a temporary workflow.json without companion files
    workflow_json = {
        'version': '0.1.0',
        'nodes': [
            {'id': 0, 'type': 'input', 'name': 'x', 'value': 1},
            {'id': 1, 'type': 'output', 'name': 'result'},
        ],
        'edges': [{'source': 0, 'target': 1, 'sourcePort': None, 'targetPort': None}],
    }

    temp_workflow_path = tmp_path / 'workflow.json'
    with open(temp_workflow_path, 'w') as f:
        json.dump(workflow_json, f)

    archive = EntryArchive()

    # Should raise FileNotFoundError for missing companion files
    with pytest.raises(FileNotFoundError) as exc_info:
        parser.parse(str(temp_workflow_path), archive, None)

    error_message = str(exc_info.value)
    assert 'Missing required companion files' in error_message
    assert 'workflow.py' in error_message
    assert 'environment.yaml' in error_message


def test_workflow_json_structure_validation(test_data_path):
    """Test that workflow.json files have the expected structure."""
    examples = ['arithmetic', 'nfdi', 'quantum_espresso']

    for example in examples:
        workflow_json_path = test_data_path / example / 'workflow.json'

        with open(workflow_json_path) as f:
            data = json.load(f)

        # Test required top-level fields
        assert 'version' in data
        assert 'nodes' in data
        assert 'edges' in data

        # Test version format
        assert isinstance(data['version'], str)
        assert data['version'] == '0.1.0'

        # Test nodes structure
        assert isinstance(data['nodes'], list)
        assert len(data['nodes']) > 0

        for node in data['nodes']:
            assert 'id' in node
            assert 'type' in node
            assert node['type'] in ['input', 'output', 'function']

            if node['type'] == 'function':
                assert 'value' in node
                assert '.' in node['value']  # Should be module.function format

        # Test edges structure
        assert isinstance(data['edges'], list)
        for edge in data['edges']:
            assert 'source' in edge
            assert 'target' in edge
            assert 'sourcePort' in edge
            assert 'targetPort' in edge


def test_companion_files_exist(test_data_path):
    """Test that all example directories have the required companion files."""
    examples = ['arithmetic', 'nfdi', 'quantum_espresso']

    for example in examples:
        example_dir = test_data_path / example

        # Check for required files
        assert (example_dir / 'workflow.json').exists()
        assert (example_dir / 'workflow.py').exists()
        assert (example_dir / 'environment.yaml').exists()


def test_get_pydantic_model(parser, test_data_path):
    """Test getting the Pydantic model from parsed workflow."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    parser.parse(str(workflow_json_path), archive, None)

    # Test getting Pydantic model
    pydantic_model = archive.workflow2.get_pydantic_model()
    assert pydantic_model is not None
    assert isinstance(pydantic_model, dict)
    assert 'version' in pydantic_model
    assert 'nodes' in pydantic_model
    assert 'edges' in pydantic_model


@pytest.mark.parametrize('example_name', ['arithmetic', 'nfdi', 'quantum_espresso'])
def test_parse_all_examples(parser, test_data_path, example_name):
    """Parameterized test to parse all workflow examples."""
    workflow_json_path = test_data_path / example_name / 'workflow.json'
    archive = EntryArchive()

    # Should not raise any exceptions
    parser.parse(str(workflow_json_path), archive, None)

    # Basic validation
    assert archive.workflow2 is not None
    assert archive.workflow2.method is not None
    assert archive.workflow2.results is not None
    assert archive.workflow2.results.n_nodes > 0
    assert archive.workflow2.results.n_edges >= 0


# ============================================================================
# Edge Parsing Tests
# ============================================================================


def test_edge_parsing_basic_structure(parser, test_data_path):
    """Test that parsing creates proper NOMAD workflow structures."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    all_tasks = get_all_tasks(workflow)

    # Verify NOMAD workflow structures are created
    assert len(workflow.inputs) > 0
    assert len(workflow.outputs) > 0
    assert len(workflow.tasks) > 0

    # Check that we have the expected number of each type
    assert len(workflow.inputs) >= ARITHMETIC_INPUTS  # x, y inputs
    assert len(workflow.outputs) >= ARITHMETIC_OUTPUTS  # result output
    assert len(all_tasks) >= ARITHMETIC_FUNCTIONS  # function tasks

    # Check that tasks have the right properties
    for task in all_tasks:
        assert task.node_type == 'function'
        assert task.node_id is not None
        assert task.module_function is not None


def test_edge_to_connection_mapping(parser, test_data_path):
    """Test that edges are correctly mapped to task input/output connections."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    # Load raw data for comparison
    with open(workflow_json_path) as f:
        raw_data = json.load(f)

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    edges = raw_data['edges']
    function_tasks = [t for t in get_all_tasks(workflow) if t.node_type == 'function']

    # Verify that each edge creates appropriate connections
    for task in function_tasks:
        node_id = task.node_id

        # Count edges targeting this node (should match input connections)
        incoming_edges = [e for e in edges if e['target'] == node_id]
        assert len(task.inputs) == len(incoming_edges)

        # Count edges sourcing from this node (should match output connections)
        outgoing_edges = [e for e in edges if e['source'] == node_id]
        # Note: outputs might be grouped by port, so we check the relationship exists
        assert len(task.outputs) > 0 if outgoing_edges else len(task.outputs) == 0


def test_port_name_mapping(parser, test_data_path):
    """Test that edge port names are correctly used in connections."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    # Load raw data for comparison
    with open(workflow_json_path) as f:
        raw_data = json.load(f)

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    edges = raw_data['edges']
    function_tasks = [t for t in get_all_tasks(workflow) if t.node_type == 'function']

    # Check first function task (get_prod_and_div)
    first_task = function_tasks[0]
    assert first_task.node_id == 0

    # Verify input port names match targetPort in edges
    incoming_edges = [e for e in edges if e['target'] == 0]
    input_port_names = {inp.name for inp in first_task.inputs}
    expected_port_names = {e['targetPort'] for e in incoming_edges}
    assert input_port_names == expected_port_names

    # Verify output port names are created for sourcePort in edges
    outgoing_edges = [e for e in edges if e['source'] == 0]
    output_port_names = {out.name for out in first_task.outputs}
    expected_source_ports = {e['sourcePort'] for e in outgoing_edges if e['sourcePort']}
    # All non-null sourcePorts should have corresponding outputs
    assert expected_source_ports.issubset(output_port_names)


def test_value_section_references(parser, test_data_path):
    """Test that task connections reference valid sections."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    all_tasks = get_all_tasks(workflow)

    # Collect all valid sections (tasks, input/output task sections)
    valid_sections = set()
    for task in all_tasks:
        valid_sections.add(task)
        # Add input/output task sections
        for input_link in task.inputs:
            if input_link.section:
                valid_sections.add(input_link.section)
        for output_link in task.outputs:
            if output_link.section:
                valid_sections.add(output_link.section)

    # Add workflow-level input/output sections
    for workflow_input in workflow.inputs:
        if workflow_input.section:
            valid_sections.add(workflow_input.section)
    for workflow_output in workflow.outputs:
        if workflow_output.section:
            valid_sections.add(workflow_output.section)

    function_tasks = [t for t in all_tasks if t.node_type == 'function']

    for task in function_tasks:
        # Check that all input connections reference valid sections
        for input_conn in task.inputs:
            assert input_conn.section is not None
            assert input_conn.section in valid_sections

        # Check that all output connections reference valid sections
        for output_conn in task.outputs:
            assert output_conn.section is not None
            assert output_conn.section in valid_sections


def test_workflow_level_inputs_outputs(parser, test_data_path):
    """Test that workflow-level inputs and outputs are properly connected."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    # Load raw data for comparison
    with open(workflow_json_path) as f:
        raw_data = json.load(f)

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    nodes = raw_data['nodes']
    input_nodes = [n for n in nodes if n['type'] == 'input']
    output_nodes = [n for n in nodes if n['type'] == 'output']

    # Workflow inputs should match input nodes
    assert len(workflow.inputs) >= len(input_nodes)

    # Workflow outputs should match output nodes
    assert len(workflow.outputs) >= len(output_nodes)

    # Each workflow input should reference a value section
    for workflow_input in workflow.inputs:
        assert workflow_input.section is not None
        # Section should be a valid PythonWorkflowDefinitionTask
        assert hasattr(workflow_input.section, 'node_type')
        assert workflow_input.section.node_type == 'input'


def test_edge_parsing_complex_workflow(parser, test_data_path):
    """Test edge parsing with the complex quantum_espresso workflow."""
    workflow_json_path = test_data_path / 'quantum_espresso' / 'workflow.json'
    archive = EntryArchive()

    # Load raw data for comparison
    with open(workflow_json_path) as f:
        raw_data = json.load(f)

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    nodes = raw_data['nodes']
    edges = raw_data['edges']

    # Verify comprehensive edge processing
    # Should have created tasks and connections
    all_tasks = get_all_tasks(workflow)
    total_entities = len(all_tasks) + len(workflow.inputs) + len(workflow.outputs)
    assert total_entities >= len(nodes)
    assert workflow.results.n_edges == len(edges)

    # Check that all function nodes have corresponding tasks
    function_nodes = [n for n in nodes if n['type'] == 'function']
    function_tasks = [t for t in all_tasks if t.node_type == 'function']
    assert len(function_tasks) == len(function_nodes)

    # Verify total connection count makes sense
    total_inputs = sum(len(task.inputs) for task in function_tasks)
    total_outputs = sum(len(task.outputs) for task in function_tasks)

    # Should have reasonable number of connections
    assert total_inputs > 0
    assert total_outputs > 0

    # Connection count should relate to edge count
    # (not exact equality due to output port grouping)
    assert total_inputs <= len(edges)  # Each edge creates at most one input
    assert total_outputs > 0  # Should have some outputs


def test_multiple_output_ports(parser, test_data_path):
    """Test handling of functions with multiple output ports."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    # Load raw data for comparison
    with open(workflow_json_path) as f:
        raw_data = json.load(f)

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    edges = raw_data['edges']

    # Find function with multiple outputs (node 0: get_prod_and_div)
    node_0_outgoing = [e for e in edges if e['source'] == 0]
    source_ports = {e['sourcePort'] for e in node_0_outgoing if e['sourcePort']}

    if len(source_ports) > 1:  # If we have multiple output ports
        # Find the corresponding task
        all_tasks = get_all_tasks(workflow)
        task_0 = next((t for t in all_tasks if t.node_id == 0), None)
        assert task_0 is not None

        # Should have output for each source port
        output_names = {out.name for out in task_0.outputs}
        assert source_ports.issubset(output_names)


def test_edge_parsing_preserves_values(parser, test_data_path):
    """Test that edge parsing preserves node values in value sections."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    # Load raw data for comparison
    with open(workflow_json_path) as f:
        raw_data = json.load(f)

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    nodes = raw_data['nodes']

    # Check that input values are preserved in input task sections
    for node in nodes:
        if node['type'] == 'input' and 'value' in node:
            # Find corresponding input section
            input_section = None
            for workflow_input in workflow.inputs:
                if (
                    workflow_input.section
                    and hasattr(workflow_input.section, 'node_id')
                    and workflow_input.section.node_id == node['id']
                ):
                    input_section = workflow_input.section
                    break
            assert input_section is not None
            # Note: We don't store the value anymore, just the structure

        if node['type'] == 'function':
            # Find corresponding task
            all_tasks = get_all_tasks(workflow)
            task = next((t for t in all_tasks if t.node_id == node['id']), None)
            assert task is not None
            assert task.module_function == node['value']


def test_edge_connection_matching_for_nomad_visualization(parser, test_data_path):
    """
    Test that each edge creates exactly matching input/output connections.

    This is critical for NOMAD's graph visualization to work correctly.
    Each edge must create:
    1. An output connection on the source task
    2. An input connection on the target task
    3. Both connections must reference the SAME section object
    """
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    # Load raw data for edge analysis
    with open(workflow_json_path) as f:
        raw_data = json.load(f)

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    edges = raw_data['edges']

    # Test all function-to-function edges (the critical ones for visualization)
    function_to_function_edges = [
        e
        for e in edges
        if any(
            t.node_id == e['source']
            for t in get_all_tasks(workflow)
            if t.node_type == 'function'
        )
        and any(
            t.node_id == e['target']
            for t in get_all_tasks(workflow)
            if t.node_type == 'function'
        )
    ]

    # Each function-to-function edge must have matching connections
    for edge in function_to_function_edges:
        source_task = next(
            (t for t in get_all_tasks(workflow) if t.node_id == edge['source']), None
        )
        target_task = next(
            (t for t in get_all_tasks(workflow) if t.node_id == edge['target']), None
        )

        assert source_task is not None, f'Source task not found for edge {edge}'
        assert target_task is not None, f'Target task not found for edge {edge}'

        # Determine expected port names
        source_port = edge.get('sourcePort')
        if source_port is None:
            source_port = 'result'
        target_port = edge.get('targetPort')

        # Find the matching output on source task
        source_output = next(
            (out for out in source_task.outputs if out.name == source_port), None
        )

        # Find the matching input on target task
        target_input = next(
            (inp for inp in target_task.inputs if inp.name == target_port), None
        )

        # Critical assertions for NOMAD graph visualization
        assert source_output is not None, (
            f"Source output '{source_port}' not found on task {edge['source']}"
        )
        assert target_input is not None, (
            f"Target input '{target_port}' not found on task {edge['target']}"
        )
        assert source_output.section is target_input.section, (
            f'Edge {edge["source"]}:{source_port} -> {edge["target"]}:{target_port} '
            f'connections reference different sections! '
            f'Source: {id(source_output.section)}, Target: {id(target_input.section)}'
        )


def test_edge_matching_quantum_espresso(parser, test_data_path):
    """
    Test edge matching on the complex quantum_espresso workflow.

    This workflow has 33 nodes and 60 edges, providing a thorough test
    of edge connection matching at scale.
    """
    workflow_json_path = test_data_path / 'quantum_espresso' / 'workflow.json'
    archive = EntryArchive()

    # Load raw data for edge analysis
    with open(workflow_json_path) as f:
        raw_data = json.load(f)

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    edges = raw_data['edges']

    # Test all function-to-function edges
    function_to_function_edges = [
        e
        for e in edges
        if any(
            t.node_id == e['source']
            for t in get_all_tasks(workflow)
            if t.node_type == 'function'
        )
        and any(
            t.node_id == e['target']
            for t in get_all_tasks(workflow)
            if t.node_type == 'function'
        )
    ]

    # Should have substantial number of function-to-function edges
    min_expected_edges = 5
    assert len(function_to_function_edges) > min_expected_edges, (
        f'Expected > {min_expected_edges} function-to-function edges, '
        f'got {len(function_to_function_edges)}'
    )

    matching_edges = 0

    # Validate each function-to-function edge
    for edge in function_to_function_edges:
        source_task = next(
            (t for t in get_all_tasks(workflow) if t.node_id == edge['source']), None
        )
        target_task = next(
            (t for t in get_all_tasks(workflow) if t.node_id == edge['target']), None
        )

        if source_task is None or target_task is None:
            continue

        # Determine expected port names
        source_port = edge.get('sourcePort')
        if source_port is None:
            source_port = 'result'
        target_port = edge.get('targetPort')

        # Find the matching connections
        source_output = next(
            (out for out in source_task.outputs if out.name == source_port), None
        )
        target_input = next(
            (inp for inp in target_task.inputs if inp.name == target_port), None
        )

        # Validate connection matching
        if (
            source_output is not None
            and target_input is not None
            and source_output.section is target_input.section
        ):
            matching_edges += 1

    # All function-to-function edges should have matching connections
    assert matching_edges == len(function_to_function_edges), (
        f'Edge matching failed: {matching_edges}/'
        f'{len(function_to_function_edges)} edges matched'
    )


def test_function_to_output_connections_regression(parser, test_data_path):
    """
    Regression test for the bug where function task outputs pointed to themselves
    instead of the correct output nodes.

    This specifically tests the arithmetic workflow where node 2 (get_square)
    should have an output that points to node 5 (output node), not back to
    the task for node 2 itself.
    """
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    # Load raw data to identify function-to-output edges
    with open(workflow_json_path) as f:
        raw_data = json.load(f)

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    edges = raw_data['edges']
    nodes = raw_data['nodes']

    # Constants for specific test case
    FUNCTION_NODE_ID = 2  # get_square function
    OUTPUT_NODE_ID = 5  # result output

    # Find edges from function nodes to output nodes
    function_to_output_edges = [
        e
        for e in edges
        if any(n['id'] == e['source'] and n['type'] == 'function' for n in nodes)
        and any(n['id'] == e['target'] and n['type'] == 'output' for n in nodes)
    ]

    # Test the specific case from arithmetic workflow:
    # Node 2 (get_square) -> Node 5 (output)
    node_2_to_5_edge = next(
        (
            e
            for e in function_to_output_edges
            if e['source'] == FUNCTION_NODE_ID and e['target'] == OUTPUT_NODE_ID
        ),
        None,
    )

    if node_2_to_5_edge is not None:
        # Find the corresponding function task
        source_task = next(
            (t for t in get_all_tasks(workflow) if t.node_id == FUNCTION_NODE_ID), None
        )
        assert source_task is not None, 'Task for node 2 not found'

        # Find the output node
        target_node = next(
            (n for n in workflow.pwd_nodes if n.node_id == OUTPUT_NODE_ID), None
        )
        assert target_node is not None, 'Output node 5 not found'

        # The critical test: task output should point to the output node,
        # not back to the task
        assert len(source_task.outputs) > 0, 'Task should have outputs'
        task_output = source_task.outputs[0]  # get_square has one output

        # This is the regression test: task output should NOT point to the
        # source task
        assert task_output.section is not source_task, (
            f'Task output incorrectly points to source task itself! '
            f'Task: {id(source_task)}, Output section: {id(task_output.section)}'
        )

        # Task output should point to the target output node
        assert task_output.section is target_node, (
            f'Task output should point to target node 5, not task 2. '
            f'Expected: {id(target_node)}, Got: {id(task_output.section)}'
        )


def test_parsing_new_quantities(parser, test_data_path):
    """
    Test parsing of the new 'working_directory' and 'output' quantities.
    Uses the 'pwd' example data which contains these fields.
    """
    # Locate the file
    workflow_json_path = test_data_path / 'data_export' / 'workflow.json'

    # Safety check: Skip if the file hasn't been created/copied yet
    if not workflow_json_path.exists():
        pytest.skip(
            f'Test data not found at {workflow_json_path}. Please create it first.'
        )

    # Parse the file
    archive = EntryArchive()
    parser.parse(str(workflow_json_path), archive, None)

    workflow = archive.workflow2
    assert workflow is not None

    # Iterate through tasks and verify our new data exists
    tasks_with_new_data = 0

    for task in get_all_tasks(workflow):
        # Check 'working_directory' (Should be a string)
        if task.working_directory:
            assert isinstance(task.working_directory, str)
            # Our mock generator used "/home/nomad/", so we check for that
            assert '/home/nomad/' in task.working_directory
            tasks_with_new_data += 1

        # Check 'output' (Should be a Dictionary/JSON)
        if task.output:
            assert isinstance(task.output, dict)

            # If it's a calculation node, it should have specific keys
            # (We check for keys we know we injected: energy, volume, structure)
            if 'energy' in task.output:
                assert isinstance(task.output['energy'], float)
            if 'volume' in task.output:
                assert isinstance(task.output['volume'], float)
            if 'structure' in task.output:
                # Structure is stored as a JSON string inside the dict
                assert isinstance(task.output['structure'], str)

    # Final Verdict: Did we actually find any data?
    assert tasks_with_new_data > 0, (
        'Parser returned 0 tasks with working_directory! Check parser.py logic.'
    )


def test_task_reference_resolution(tmp_path):
    """
    Test that _resolve_task_references correctly:
    1. Handles absolute paths (strips them).
    2. Finds local files.
    3. Converts Tasks to TaskReferences.
    4. Preserves inputs/outputs.
    """
    # 1. Setup a mock file system in the temp directory
    # Structure:
    # /tmp/workflow.json
    # /tmp/mock_calc/pw.out

    # Create the calculation folder and dummy output file
    calc_folder = tmp_path / 'mock_calc'
    calc_folder.mkdir()
    (calc_folder / 'pw.out').write_text('dummy content')

    # Define the mainfile path
    mainfile_path = str(tmp_path / 'workflow.json')

    # 2. Create a Mock Workflow with one task pointing to that folder
    workflow = PythonWorkflowDefinition()

    # Create a generic PWD task with an absolute path
    task = PythonWorkflowDefinitionTask()
    task.name = 'My Simulation'
    task.working_directory = '/home/user/absolute/path/to/mock_calc'

    # Add some dummy inputs/outputs to ensure they are preserved
    # Note: In a real scenario these would be Link objects, but for this test
    # simply checking that the list content is preserved is enough.
    task.inputs = []
    task.outputs = []

    workflow.tasks.append(task)

    # 3. Mock the Archive object
    # The parser needs archive.metadata.mainfile to build the relative link
    mock_archive = MagicMock()
    mock_archive.metadata.mainfile = 'upload/workflow.json'

    # 4. Run the resolution
    parser = PythonWorkflowDefinitionParser()
    logger = MagicMock()

    parser._resolve_task_references(workflow, mainfile_path, mock_archive, logger)

    # 5. Assertions
    assert len(workflow.tasks) == 1
    result_task = workflow.tasks[0]

    # It should now be a TaskReference, not a PythonWorkflowDefinitionTask
    assert isinstance(result_task, TaskReference)

    # The name should be preserved
    assert result_task.name == 'My Simulation'

    # The link should be correct (relative to upload root)
    # Expected:
    #  ../upload/archive/mainfile/{dir_of_json}/{clean_folder}/{file}#/workflow2
    expected_link = '../upload/archive/mainfile/upload/mock_calc/pw.out#/workflow2'

    # MProxy objects string representation wraps the value (e.g. MProxy(value))
    # We check if the expected link is inside the string representation
    assert expected_link in str(result_task.task)

def test_workflow_simplification(parser, test_data_path):
    """
    Test that the workflow graph is simplified by grouping utility tasks
    (tasks without working_directory AND low connectivity) into a
    'Utility Functions' sub-workflow.
    """
    # Locate the file (using data_export as it contains the working_directory fields)
    workflow_json_path = test_data_path / 'data_export' / 'workflow.json'

    # Safety check consistent with other tests
    if not workflow_json_path.exists():
        pytest.skip(
            f'Test data not found at {workflow_json_path}. Please create it first.'
        )

    # Parse the file
    archive = EntryArchive()
    parser.parse(str(workflow_json_path), archive, None)

    workflow = archive.workflow2
    assert workflow is not None

    # 1. Inspect the Top-Level Tasks
    top_level_tasks = workflow.tasks

    # Identify the sub-workflow
    utility_workflows = [t for t in top_level_tasks if t.name == 'Utility Functions']

    assert len(utility_workflows) == 1, (
        f"Expected 1 'Utility Functions' sub-workflow, found {len(utility_workflows)}"
    )

    # Identify Main Tasks at top level
    main_tasks = [t for t in top_level_tasks if t.name != 'Utility Functions']

    # We identify "Hubs" by name here based on the known dataset topology.
    # In this dataset, 'get_list' and 'generate_structures' are high-degree nodes.
    known_hubs = [
        'Function python_workflow_definition.shared.get_list',
        'Function workflow.generate_structures',
    ]

    for task in main_tasks:
        # Check if it is a file producer
        has_files = hasattr(task, 'working_directory') and task.working_directory

        # Check if it is a known hub
        is_hub = task.name in known_hubs

        assert has_files or is_hub, (
            f"Top level task '{task.name}' is neither a file producer nor a known Hub. "
            f'It should have been moved to Utility Functions.'
        )

    # 3. Verify Utility Workflow Content
    utility_workflow = utility_workflows[0]

    # Check that it contains tasks
    assert len(utility_workflow.tasks) > 0, 'Utility workflow should not be empty'

    # Verify tasks inside are actually utilities
    for task in utility_workflow.tasks:
        # Must not have working directory
        if hasattr(task, 'working_directory'):
            assert not task.working_directory, (
                f'Task {task.name} in utility workflow has working_directory'
            )

        # Must not be a major hub
        assert task.name not in known_hubs, (
            f"Hub task '{task.name}' was incorrectly moved to Utility Functions."
        )

    # 4. Verify specific counts based on the sample json
    # (Nodes 0, 1, 2, 4, 5, 8, 10, 12, 14, 16 have working directories -> 10 Main)
    # (Nodes 3, 6, 7, 9, 11, 13, 15 have null -> 7 Utility)
    count_main_tasks = 10
    count_utility_tasks = 7
    assert len(main_tasks) == count_main_tasks, (
        f'Expected {count_main_tasks} main tasks (Producers + Hubs),\
              got {len(main_tasks)}'
    )
    assert len(utility_workflow.tasks) == count_utility_tasks, (
        f'Expected {count_utility_tasks} utility tasks,\
              got {len(utility_workflow.tasks)}'
    )

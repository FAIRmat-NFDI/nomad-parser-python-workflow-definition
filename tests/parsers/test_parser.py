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

import pytest

from nomad.datamodel import EntryArchive
from nomad_parser_pwd.parsers.parser import PythonWorkflowDefinitionParser


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
    """Test that edge parsing creates proper workflow value sections."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    # Verify workflow values are created
    assert len(workflow.workflow_values) > 0

    # Check that we have value sections for all node types
    input_values = [v for v in workflow.workflow_values if v.node_type == 'input']
    output_values = [v for v in workflow.workflow_values if v.node_type == 'output']
    function_values = [v for v in workflow.workflow_values if v.node_type == 'function']

    assert len(input_values) >= ARITHMETIC_INPUTS  # x, y inputs
    assert len(output_values) >= ARITHMETIC_OUTPUTS  # result output
    # function nodes + their outputs
    assert len(function_values) >= ARITHMETIC_FUNCTIONS


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
    function_tasks = [t for t in workflow.workflow_tasks if t.node_type == 'function']

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
    function_tasks = [t for t in workflow.workflow_tasks if t.node_type == 'function']

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
    """Test that task connections reference the correct value sections."""
    workflow_json_path = test_data_path / 'arithmetic' / 'workflow.json'
    archive = EntryArchive()

    parser.parse(str(workflow_json_path), archive, None)
    workflow = archive.workflow2

    # Get node ID to value section mapping (for validation)
    # node_to_value = {v.node_id: v for v in workflow.workflow_values}

    function_tasks = [t for t in workflow.workflow_tasks if t.node_type == 'function']

    for task in function_tasks:
        # Check that all input connections reference valid value sections
        for input_conn in task.inputs:
            assert input_conn.section is not None
            assert input_conn.section in workflow.workflow_values

        # Check that all output connections reference valid value sections
        for output_conn in task.outputs:
            assert output_conn.section is not None
            assert output_conn.section in workflow.workflow_values


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
        assert workflow_input.section in workflow.workflow_values


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
    # Should have additional output port values
    assert len(workflow.workflow_values) > len(nodes)
    assert workflow.results.n_edges == len(edges)

    # Check that all function nodes have corresponding tasks
    function_nodes = [n for n in nodes if n['type'] == 'function']
    function_tasks = [t for t in workflow.workflow_tasks if t.node_type == 'function']
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
        task_0 = next((t for t in workflow.workflow_tasks if t.node_id == 0), None)
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

    # Check that input values are preserved
    for node in nodes:
        if node['type'] == 'input' and 'value' in node:
            # Find corresponding value section
            value_section = next(
                (v for v in workflow.workflow_values if v.node_id == node['id']), None
            )
            assert value_section is not None
            assert value_section.value == node['value']

        if node['type'] == 'function':
            # Find corresponding value section
            value_section = next(
                (v for v in workflow.workflow_values if v.node_id == node['id']), None
            )
            assert value_section is not None
            assert value_section.value == node['value']


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
            for t in workflow.workflow_tasks
            if t.node_type == 'function'
        )
        and any(
            t.node_id == e['target']
            for t in workflow.workflow_tasks
            if t.node_type == 'function'
        )
    ]

    # Each function-to-function edge must have matching connections
    for edge in function_to_function_edges:
        source_task = next(
            (t for t in workflow.workflow_tasks if t.node_id == edge['source']), None
        )
        target_task = next(
            (t for t in workflow.workflow_tasks if t.node_id == edge['target']), None
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
        assert (
            source_output is not None
        ), f"Source output '{source_port}' not found on task {edge['source']}"
        assert (
            target_input is not None
        ), f"Target input '{target_port}' not found on task {edge['target']}"
        assert source_output.section is target_input.section, (
            f"Edge {edge['source']}:{source_port} -> {edge['target']}:{target_port} "
            f"connections reference different sections! "
            f"Source: {id(source_output.section)}, Target: {id(target_input.section)}"
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
            for t in workflow.workflow_tasks
            if t.node_type == 'function'
        )
        and any(
            t.node_id == e['target']
            for t in workflow.workflow_tasks
            if t.node_type == 'function'
        )
    ]

    # Should have substantial number of function-to-function edges
    assert (
        len(function_to_function_edges) > 5
    ), f'Expected > 5 function-to-function edges, got {len(function_to_function_edges)}'

    matching_edges = 0

    # Validate each function-to-function edge
    for edge in function_to_function_edges:
        source_task = next(
            (t for t in workflow.workflow_tasks if t.node_id == edge['source']), None
        )
        target_task = next(
            (t for t in workflow.workflow_tasks if t.node_id == edge['target']), None
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
    assert (
        matching_edges == len(function_to_function_edges)
    ), f'Edge matching failed: {matching_edges}/{len(function_to_function_edges)} edges matched'

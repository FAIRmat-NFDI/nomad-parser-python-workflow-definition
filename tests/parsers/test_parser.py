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
        str(workflow_json_path),
        'application/json',
        buffer,
        decoded_buffer
    )
    
    assert result is True


def test_is_mainfile_non_workflow_json(parser):
    """Test is_mainfile rejects non-workflow JSON files."""
    # Test with a non-workflow.json filename
    result = parser.is_mainfile(
        'random_file.json',
        'application/json',
        b'{"version": "1.0"}',
        '{"version": "1.0"}'
    )
    
    assert result is False


def test_is_mainfile_invalid_json_structure(parser):
    """Test is_mainfile rejects invalid JSON structure."""
    result = parser.is_mainfile(
        'workflow.json',
        'application/json',
        b'{"invalid": "structure"}',
        '{"invalid": "structure"}'
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
    
    # Test method section
    assert archive.workflow2.method is not None
    assert archive.workflow2.method.version == '0.1.0'
    assert archive.workflow2.method.workflow_definition is not None
    
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
        "version": "0.1.0",
        "nodes": [
            {"id": 0, "type": "input", "name": "x", "value": 1},
            {"id": 1, "type": "output", "name": "result"}
        ],
        "edges": [
            {"source": 0, "target": 1, "sourcePort": None, "targetPort": None}
        ]
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


@pytest.mark.parametrize("example_name", ["arithmetic", "nfdi", "quantum_espresso"])
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

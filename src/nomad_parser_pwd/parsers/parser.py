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
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.config import config
from nomad.parsing.parser import MatchingParser
from python_workflow_definition.models import PythonWorkflowDefinitionWorkflow

from ..schema_packages.pwd import (
    PythonWorkflowDefinition,
)

configuration = config.get_plugin_entry_point(
    'nomad_parser_pwd.parsers:parser_entry_point'
)


class PythonWorkflowDefinitionParser(MatchingParser):
    """
    Parser for Python Workflow Definition JSON files.

    This parser identifies and processes JSON files containing Python workflow
    definitions as defined by the python-workflow-definition package. The parser
    validates the JSON structure against the expected schema and creates NOMAD
    workflow archives.
    """

    def __init__(self):
        super().__init__(
            name='parsers/python_workflow_definition',
            code_name='Python Workflow Definition',
            code_homepage='https://github.com/nomad-coe/python-workflow-definition',
            mainfile_name_re=r'(^|.*/)workflow\.json$',
            mainfile_mime_re=r'application/json',
        )

    def is_mainfile(
        self,
        filename: str,
        mime: str,
        buffer: bytes,
        decoded_buffer: str,
        compression: str = None,
    ) -> bool:
        """
        Determine if the file is a Python workflow definition JSON file.

        Requirements for detection:
        1. File must be named exactly "workflow.json"
        2. The JSON content must match PWD structure

        Note: Companion files (workflow.py, environment.yaml) are checked during
        parsing and will cause parsing errors if missing, but don't prevent
        initial file detection for partial parsing scenarios.

        Args:
            filename: Name of the file being tested
            mime: MIME type of the file
            buffer: Raw file content as bytes
            decoded_buffer: File content as decoded string
            compression: Compression type (if any)

        Returns:
            True if this is a Python workflow definition file, False otherwise
        """
        # Check if filename is exactly "workflow.json"
        if not filename.endswith('/workflow.json') and filename != 'workflow.json':
            return False

        # Only validate JSON structure for initial detection
        # Companion files will be checked during parsing
        return self._validate_pwd_structure(decoded_buffer)

    def _check_required_companion_files(self, json_filename: str, logger) -> None:
        """
        Check for required companion files and raise errors if missing.

        This allows partial parsing but marks parsing as unsuccessful if
        companion files are missing.
        """
        import os

        directory = os.path.dirname(json_filename)
        if not directory:
            directory = '.'

        missing_files = []

        # Check for required workflow.py file
        workflow_py_path = os.path.join(directory, 'workflow.py')
        if not os.path.isfile(workflow_py_path):
            missing_files.append('workflow.py')

        # Check for required environment.yaml file
        environment_yaml_path = os.path.join(directory, 'environment.yaml')
        if not os.path.isfile(environment_yaml_path):
            missing_files.append('environment.yaml')

        # Log status of companion files
        self._log_companion_files(json_filename, logger)

        # Raise error if any required files are missing
        if missing_files:
            error_msg = (
                f'Missing required companion files: {", ".join(missing_files)}. '
                'Python Workflow Definition projects require workflow.json, '
                'workflow.py, and environment.yaml files.'
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info('All required companion files found')

    def _has_companion_files(self, json_filename: str) -> bool:
        """
        Check for required companion files.

        Required: workflow.py, environment.yaml
        """
        import os

        # Get the directory containing the workflow.json file
        directory = os.path.dirname(json_filename)
        if not directory:
            directory = '.'

        # Check for required workflow.py file
        workflow_py_path = os.path.join(directory, 'workflow.py')
        if not os.path.isfile(workflow_py_path):
            return False

        # Check for required environment.yaml file
        environment_yaml_path = os.path.join(directory, 'environment.yaml')
        if not os.path.isfile(environment_yaml_path):
            return False

        return True  # Both required files are present

    def _validate_pwd_structure(self, content: str) -> bool:
        """Validate if content matches Python workflow definition structure."""
        try:
            data = json.loads(content)

            # Check basic structure
            if not self._has_required_fields(data):
                return False

            # Check for PWD-specific patterns
            if not self._has_pwd_patterns(data):
                return False

            # Validate nodes and edges structure
            if not self._validate_nodes_and_edges(data):
                return False

            # Final validation with Pydantic model
            try:
                PythonWorkflowDefinitionWorkflow.load_json_str(content)
                return True
            except Exception:
                return False

        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def _has_pwd_patterns(self, data: dict) -> bool:
        """Check for Python workflow definition specific patterns."""
        # Check version format (semantic versioning)
        version = data.get('version', '')
        if not isinstance(version, str) or not version:
            return False

        # Look for function nodes with module.function pattern
        nodes = data.get('nodes', [])
        function_nodes = [n for n in nodes if n.get('type') == 'function']

        if function_nodes:
            # Check if at least one function node has module.function format
            has_module_function = any(
                isinstance(node.get('value'), str) and '.' in node.get('value', '')
                for node in function_nodes
            )
            if not has_module_function:
                return False

        # Check edge structure with sourcePort/targetPort
        edges = data.get('edges', [])
        if edges:
            # Check if edges have the expected PWD structure
            has_port_structure = any(
                'sourcePort' in edge and 'targetPort' in edge for edge in edges
            )
            if not has_port_structure:
                return False

        return True

    def _has_required_fields(self, data: dict) -> bool:
        """Check if data has required top-level fields."""
        if not isinstance(data, dict):
            return False
        required_fields = ['version', 'nodes', 'edges']
        return all(field in data for field in required_fields)

    def _validate_nodes_and_edges(self, data: dict) -> bool:
        """Validate nodes and edges structure."""
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])

        if not isinstance(nodes, list) or not isinstance(edges, list):
            return False

        # Validate node structure
        for node in nodes:
            if not self._validate_node(node):
                return False

        # Validate edge structure
        for edge in edges:
            if not self._validate_edge(edge):
                return False

        return True

    def _validate_node(self, node: dict) -> bool:
        """Validate individual node structure."""
        if not isinstance(node, dict):
            return False
        if not all(field in node for field in ['id', 'type']):
            return False
        if node.get('type') not in ['input', 'output', 'function']:
            return False
        return True

    def _validate_edge(self, edge: dict) -> bool:
        """Validate individual edge structure."""
        if not isinstance(edge, dict):
            return False
        if not all(field in edge for field in ['source', 'target']):
            return False
        return True

    def _log_companion_files(self, json_filename: str, logger) -> None:
        """Log information about companion files in the directory."""
        import os

        directory = os.path.dirname(json_filename)
        if not directory:
            directory = '.'

        # Check for workflow.py (required)
        workflow_py_path = os.path.join(directory, 'workflow.py')
        has_workflow_py = os.path.isfile(workflow_py_path)

        # Check for environment.yaml (required)
        environment_yaml_path = os.path.join(directory, 'environment.yaml')
        has_environment_yaml = os.path.isfile(environment_yaml_path)

        logger.info(
            f'Required files - workflow.py: {has_workflow_py}, '
            f'environment.yaml: {has_environment_yaml}'
        )

        if has_workflow_py and has_environment_yaml:
            logger.info('All required companion files found')
        else:
            missing_files = []
            if not has_workflow_py:
                missing_files.append('workflow.py')
            if not has_environment_yaml:
                missing_files.append('environment.yaml')
            logger.warning(f'Missing required files: {", ".join(missing_files)}')

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        """
        Parse a Python workflow definition JSON file and populate the archive.

        Args:
            mainfile: Path to the main file to be parsed
            archive: The archive object to populate with parsed data
            logger: Logger instance for reporting parsing progress/issues
            child_archives: Dict of child archives (not used for PWD files)
        """
        # Handle case where logger is None (e.g., in tests)
        if logger is None:
            from nomad.utils import get_logger

            logger = get_logger(__name__)

        logger.info('Starting Python Workflow Definition parser')

        try:
            # Check for required companion files first
            self._check_required_companion_files(mainfile, logger)

            # Read and validate the JSON file
            with open(mainfile, encoding='utf-8') as f:
                file_content = f.read()

            logger.info(f'Read workflow definition file: {mainfile}')

            # Check for companion files and log their presence
            self._log_companion_files(mainfile, logger)

            # Parse using Pydantic model for validation
            try:
                # We retain this to ensure the file is valid PWD format
                PythonWorkflowDefinitionWorkflow.load_json_str(file_content)
                logger.info('Successfully validated workflow definition structure')

                # Create the NOMAD workflow
                workflow = PythonWorkflowDefinition()

                # Set workflow name from filename if not provided
                if not workflow.name:
                    base_name = os.path.splitext(os.path.basename(mainfile))[0]
                    workflow.name = f'Python Workflow Definition: {base_name}'

                # Validate using the Pydantic model
                data = PythonWorkflowDefinitionWorkflow.load_json_str(file_content)

                # Load into the NOMAD section
                workflow.load_from_pydantic_model(data)

                # Set the workflow in the archive
                archive.workflow2 = workflow

                # Trigger normalization
                workflow.normalize(archive, logger)

                logger.info('Successfully created Python Workflow Definition from JSON')

                # Log workflow statistics from the NOMAD structures
                logger.info(f'Created workflow with {len(workflow.tasks)} tasks')
                logger.info(f'Workflow inputs: {len(workflow.inputs)}')
                logger.info(f'Workflow outputs: {len(workflow.outputs)}')

                # Log task details
                for i, task in enumerate(workflow.tasks):
                    logger.info(
                        f'Task {i}: {task.name} '
                        f'(node_id={task.node_id}, type={task.node_type})'
                    )

            except Exception as e:
                logger.error(f'Failed to validate workflow definition: {e}')
                raise

            except Exception as e:
                logger.error(f'Failed to validate workflow definition: {e}')
                raise

        except FileNotFoundError:
            logger.error(f'Could not find file: {mainfile}')
            raise
        except json.JSONDecodeError as e:
            logger.error(f'Invalid JSON in file {mainfile}: {e}')
            raise
        except Exception as e:
            logger.error(f'Error parsing file {mainfile}: {e}')
            raise


# Create an alias for backwards compatibility and consistency with other parsers
NewParser = PythonWorkflowDefinitionParser

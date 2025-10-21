from nomad.config.models.plugins import ParserEntryPoint
from pydantic import Field


class PythonWorkflowDefinitionParserEntryPoint(ParserEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_parser_pwd.parsers.parser import PythonWorkflowDefinitionParser

        return PythonWorkflowDefinitionParser()


parser_entry_point = PythonWorkflowDefinitionParserEntryPoint(
    name='PythonWorkflowDefinitionParser',
    description='Parser for Python Workflow Definition JSON files.',
    mainfile_name_re=r'.*/workflow\.json$',
    mainfile_mime_re=r'application/json',
)

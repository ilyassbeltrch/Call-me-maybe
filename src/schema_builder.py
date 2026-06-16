"""Build JSON schema from function definitions."""

import json
from typing import Any, Optional


class SchemaBuilder:
    """Builds JSON schema constraints from function definitions."""

    def __init__(self, functions_definition: list[dict[str, Any]]):
        """Initialize with function definitions.

        Args:
            functions_definition: List of function definitions.
        """
        self.functions = functions_definition

    def get_output_schema(self) -> dict[str, Any]:
        """Get the expected output JSON schema.

        Returns:
            Schema defining the structure of function calls.
        """
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [f["name"] for f in self.functions],
                },
                "parameters": {
                    "type": "object",
                    "properties": self._build_all_parameters(),
                },
            },
            "required": ["name", "parameters"],
        }

    def _build_all_parameters(self) -> dict[str, Any]:
        """Build parameter schema for all functions."""
        params = {}
        for func in self.functions:
            for param_name, param_info in func.get("parameters", {}).items():
                if param_name not in params:
                    param_type = param_info.get("type", "string")
                    if param_type == "number":
                        params[param_name] = {"type": "number"}
                    elif param_type == "boolean":
                        params[param_name] = {"type": "boolean"}
                    else:
                        params[param_name] = {"type": "string"}
        return params

    def get_function_by_name(self, name: str) -> Optional[dict[str, Any]]:
        """Get function definition by name.

        Args:
            name: Function name

        Returns:
            Function definition or None if not found.
        """
        for func in self.functions:
            if func["name"] == name:
                return func
        return None

    def get_required_params(self, function_name: str) -> list[str]:
        """Get required parameter names for a function.

        Args:
            function_name: Name of the function

        Returns:
            List of required parameter names.
        """
        func = self.get_function_by_name(function_name)
        if not func:
            return []
        return list(func.get("parameters", {}).keys())

    def get_param_type(
        self, function_name: str, param_name: str
    ) -> Optional[str]:
        """Get the expected type for a parameter.

        Args:
            function_name: Name of the function
            param_name: Name of the parameter

        Returns:
            Type string or None if not found.
        """
        func = self.get_function_by_name(function_name)
        if not func:
            return None
        
        params = func.get("parameters", {})
        if param_name not in params:
            return None
        
        return params[param_name].get("type", "string")

import json
import sys
from typing import Any, List, cast
from pydantic import ValidationError
from src import parsing


class Loader:
    """Loads and validates input JSON files using Pydantic models."""

    def load_functions(self, path: str) -> List[parsing.FunctionDef]:
        data = self._read_json(path)
        functions: List[parsing.FunctionDef] = []
        for item in data:
            try:
                functions.append(parsing.FunctionDef.model_validate(item))
            except ValidationError as e:
                sys.exit(f"Error: invalid function definition {item}: {e}")
        return functions

    def load_prompts(self, path: str) -> List[parsing.FunctionCall]:
        data = self._read_json(path)
        prompts: List[parsing.FunctionCall] = []
        for item in data:
            try:
                prompts.append(parsing.FunctionCall.model_validate(item))
            except ValidationError as e:
                sys.exit(f"Error: invalid prompt entry {item}: {e}")
        return prompts

    def _read_json(self, path: str) -> list[Any]:
        try:
            with open(path, "r") as f:
                return cast(list[Any], json.load(f))
        except FileNotFoundError:
            sys.exit(f"Error: file not found: {path}")
        except json.JSONDecodeError as e:
            sys.exit(f"Error: invalid JSON in {path}: {e}")

from pydantic import BaseModel
from typing import Dict, Union


class Parameter(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter


class FunctionCall(BaseModel):
    prompt: str
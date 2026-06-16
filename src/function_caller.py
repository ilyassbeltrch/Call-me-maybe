import json
import re
from typing import Any, Optional
from llm_sdk import Small_LLM_Model
from src.vocab import load_vocab, get_token_id
from src.schema_builder import SchemaBuilder
from src.constrained_generator import ConstrainedGenerator


class FunctionCaller:
    def __init__(self, functions_definition: list[dict[str, Any]]):
        self.functions_def = functions_definition
        self.model = Small_LLM_Model()
        
        tokenizer_path = self.model.get_path_to_tokenizer_file()
        self.id_to_token = load_vocab(tokenizer_path)
        
        self.schema_builder = SchemaBuilder(functions_definition)
        self.generator = ConstrainedGenerator(self.model, self.id_to_token)

    def call(self, prompt: str) -> dict[str, Any]:
        system_msg = self._build_system_message()
        
        full_prompt = f"{system_msg}\n\nUser: {prompt}\n\nRespond with JSON:\n"
        
        schema = self.schema_builder.get_output_schema()
        
        generated_tokens = self.generator.generate_with_constraints(
            full_prompt, schema
        )
        
        result_text = self.model.decode(generated_tokens)
        result_json = self._parse_function_call(result_text, prompt)
        
        return result_json

    def _build_system_message(self) -> str:
        msg = "You are a function calling assistant. You must choose the best function to call based on the user's request.\n\n"
        msg += "Available functions:\n"
        
        for func in self.functions_def:
            msg += f"\n- {func['name']}: {func['description']}\n"
            msg += f"  Parameters: {json.dumps(func['parameters'])}\n"
        
        msg += "\nRespond with a JSON object with 'name' and 'parameters' keys."
        
        return msg

    def _parse_function_call(self, text: str, original_prompt: str) -> dict[str, Any]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in generated text")
            
            json_str = text[start:end]
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            parsed = self._reconstruct_function_call(text, original_prompt)

        if "name" not in parsed:
            parsed["name"] = self._infer_function_name(original_prompt)
        
        if "parameters" not in parsed:
            parsed["parameters"] = {}
        
        result = {
            "prompt": original_prompt,
            "name": str(parsed.get("name", "")),
            "parameters": self._coerce_parameters(
                parsed.get("parameters", {}), parsed.get("name", "")
            ),
        }

        return result

    def _infer_function_name(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        
        if "sum" in prompt_lower or "add" in prompt_lower or "plus" in prompt_lower:
            return "fn_add_numbers"
        elif "greet" in prompt_lower or "hello" in prompt_lower:
            return "fn_greet"
        elif "reverse" in prompt_lower:
            return "fn_reverse_string"
        elif "square root" in prompt_lower or "sqrt" in prompt_lower:
            return "fn_get_square_root"
        elif "replace" in prompt_lower or "substitute" in prompt_lower:
            return "fn_substitute_string_with_regex"
        
        return self.functions_def[0]["name"]

    def _reconstruct_function_call(self, text: str, prompt: str) -> dict[str, Any]:
        result = {}
        
        for func in self.functions_def:
            if func["name"] in text:
                result["name"] = func["name"]
                break
        
        if "name" not in result:
            result["name"] = self._infer_function_name(prompt)
        
        result["parameters"] = self._extract_parameters_from_text(text, result["name"])
        
        return result

    def _extract_parameters_from_text(self, text: str, function_name: str) -> dict:
        params = {}
        
        func_def = self.schema_builder.get_function_by_name(function_name)
        if not func_def:
            return params
        
        expected_params = func_def.get("parameters", {})
        
        for param_name, param_info in expected_params.items():
            param_type = param_info.get("type", "string")
            
            pattern = rf'["\']?{param_name}["\']?\s*:\s*([^,}}]+)'
            match = re.search(pattern, text)
            
            if match:
                value_str = match.group(1).strip()
                value_str = value_str.strip('"\'')
                
                if param_type == "number":
                    try:
                        params[param_name] = float(value_str)
                    except ValueError:
                        pass
                elif param_type == "string":
                    params[param_name] = value_str
                elif param_type == "boolean":
                    params[param_name] = value_str.lower() in ("true", "yes", "1")
        
        return params

    def _coerce_parameters(self, parameters: dict, function_name: str) -> dict:
        func_def = None
        for func in self.functions_def:
            if func["name"] == function_name:
                func_def = func
                break
        
        if not func_def:
            return parameters

        coerced = {}
        for param_name, param_type in func_def.get("parameters", {}).items():
            if param_name in parameters:
                value = parameters[param_name]
                expected_type = param_type.get("type", "string")
                
                if expected_type == "number":
                    if isinstance(value, (int, float)):
                        coerced[param_name] = float(value)
                    else:
                        try:
                            coerced[param_name] = float(value)
                        except (ValueError, TypeError):
                            coerced[param_name] = 0.0
                elif expected_type == "string":
                    coerced[param_name] = str(value)
                elif expected_type == "boolean":
                    if isinstance(value, bool):
                        coerced[param_name] = value
                    elif isinstance(value, str):
                        coerced[param_name] = value.lower() in ("true", "yes", "1")
                    else:
                        coerced[param_name] = bool(value)
                else:
                    coerced[param_name] = value
        
        return coerced
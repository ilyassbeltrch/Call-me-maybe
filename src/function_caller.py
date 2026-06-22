# src/function_caller.py
import numpy as np
from typing import Any
from llm_sdk import Small_LLM_Model
from src.vocab import load_vocab
from src.state_machine import JSONStateMachine, State


class FunctionCaller:
    """Picks a function and arguments for a prompt using ONLY constrained
    decoding — the LLM picks everything, we only mask invalid tokens."""

    def __init__(self, functions_definition: list[dict[str, Any]]) -> None:
        self.functions_def = functions_definition
        self.model = Small_LLM_Model()

        tokenizer_path = self.model.get_path_to_tokenizer_file()
        self.id_to_str = load_vocab(tokenizer_path)

        self.function_names = [f["name"] for f in functions_definition]

    def call(self, prompt: str) -> dict[str, Any]:
       
        full_prompt = self._build_prompt(prompt)
        prompt_ids: list[int] = self.model.encode(full_prompt).tolist()[0]

        
        chosen_name = self._generate_function_name(prompt_ids)

        func_def = next(f for f in self.functions_def if f["name"] == chosen_name)
        param_names = list(func_def["parameters"].keys())
        param_types = {
            k: v.get("type", "string") for k, v in func_def["parameters"].items()
        }

        
        sm = JSONStateMachine(
            id_to_str=self.id_to_str,
            function_names=[chosen_name],  
            param_names=param_names,
            param_types=param_types,
        )

        generated_ids = self._run_state_machine(prompt_ids, sm)
        json_text = self.model.decode(generated_ids)

        import json as json_lib
        parsed = json_lib.loads(json_text)

        return {
            "prompt": prompt,
            "name": parsed["name"],
            "parameters": parsed["parameters"],
        }

    def _build_prompt(self, user_prompt: str) -> str:
        lines = ["Available functions:"]
        for f in self.functions_def:
            lines.append(f"- {f['name']}: {f['description']}")
        lines.append(f"\nUser request: {user_prompt}")
        lines.append("Function to call:")
        return "\n".join(lines)

    def _generate_function_name(self, prompt_ids: list[int]) -> str:
        """Run a tiny state machine that ONLY picks the function name,
        with no surrounding JSON — just to decide which function fits."""
        current_ids = prompt_ids.copy()
        built = ""

        while built not in self.function_names:
            logits = self.model.get_logits_from_input_ids(current_ids)
            valid_ids = self._tokens_continuing_any(self.function_names, built)
            if not valid_ids:
                break
            masked = self._mask(logits, valid_ids)
            next_id = int(np.argmax(masked))
            current_ids.append(next_id)
            built += self.id_to_str.get(next_id, "")

        return built

    def _tokens_continuing_any(self, targets: list[str], built: str) -> list[int]:
        valid: set[int] = set()
        for target in targets:
            if not target.startswith(built):
                continue
            remaining = target[len(built):]
            for token_id, token_str in self.id_to_str.items():
                if token_str and remaining.startswith(token_str):
                    valid.add(token_id)
        return list(valid)

    def _run_state_machine(
        self, prompt_ids: list[int], sm: JSONStateMachine, max_tokens: int = 200
    ) -> list[int]:
        current_ids = prompt_ids.copy()
        generated: list[int] = []

        for _ in range(max_tokens):
            valid_ids = sm.get_valid_ids()
            if not valid_ids:
                break

            logits = self.model.get_logits_from_input_ids(current_ids)
            masked = self._mask(logits, valid_ids)
            next_id = int(np.argmax(masked))

            current_ids.append(next_id)
            generated.append(next_id)
            sm.update(next_id)

            if sm.state == State.DONE:
                final_ids = sm.get_valid_ids()
                if final_ids:
                    generated.append(final_ids[0])
                break

        return generated

    def _mask(self, logits: list[float], valid_ids: list[int]) -> np.ndarray:
        arr = np.array(logits, dtype=np.float32)
        masked = np.full_like(arr, fill_value=-np.inf)
        for i in valid_ids:
            masked[i] = arr[i]
        return masked
import json
import re
from typing import Any, Optional
import numpy as np
from llm_sdk import Small_LLM_Model


class ConstrainedGenerator:
    def __init__(self, model: Small_LLM_Model, id_to_token: dict[int, str]):
        self.model = model
        self.id_to_token = id_to_token
        self.token_to_id = {v: k for k, v in id_to_token.items()}

    def generate_with_constraints(
        self, prompt: str, schema: dict[str, Any], max_tokens: int = 256
    ) -> list[int]:
        prompt_tensor = self.model.encode(prompt)
        current_ids: list[int] = prompt_tensor.tolist()[0]
        generated: list[int] = []

        json_state = {
            "depth": 0,
            "in_object": False,
            "in_string": False,
            "last_key": None,
            "keys_used": set(),
            "structure": [],
        }

        for step in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(current_ids)

            valid_ids = self._get_valid_tokens(json_state, step)

            masked = self._mask_logits(logits, valid_ids)

            next_id = int(np.argmax(masked))

            if next_id not in valid_ids:
                best_score = -np.inf
                best_id = valid_ids[0] if valid_ids else 0
                for vid in valid_ids:
                    if logits[vid] > best_score:
                        best_score = logits[vid]
                        best_id = vid
                next_id = best_id

            token_str = self.id_to_token.get(next_id, "")
            if token_str in ["<|im_end|>", "<|end|>", "</s>"]:
                break

            current_ids.append(next_id)
            generated.append(next_id)

            self._update_json_state(json_state, token_str, next_id)

            if self._is_complete_json(json_state, generated):
                break

        return generated

    def _get_valid_tokens(self, json_state: dict, step: int) -> list[int]:
        valid = []

        if step == 0:
            if "{" in self.token_to_id:
                valid.append(self.token_to_id["{"])
            if " " in self.token_to_id:
                valid.append(self.token_to_id[" "])
        elif not json_state["in_string"]:
            if json_state["depth"] == 0:
                if "{" in self.token_to_id:
                    valid.append(self.token_to_id["{"])
            else:
                valid_chars = [",", "}", '"', " ", ":"]
                for char in valid_chars:
                    if char in self.token_to_id:
                        valid.append(self.token_to_id[char])
        else:
            for tid in range(min(1000, len(self.id_to_token))):
                valid.append(tid)
            if '"' in self.token_to_id:
                valid.append(self.token_to_id['"'])

        for char in ["{", "}", "[", "]", '"', ":", ",", " "]:
            if char in self.token_to_id:
                valid.append(self.token_to_id[char])

        return list(set(valid))

    def _mask_logits(self, logits: list[float], valid_ids: list[int]) -> np.ndarray:
        arr = np.array(logits, dtype=np.float32)
        masked = np.full_like(arr, fill_value=-np.inf)

        for vid in valid_ids:
            if 0 <= vid < len(arr):
                masked[vid] = arr[vid]

        return masked

    def _update_json_state(self, state: dict, token_str: str, token_id: int) -> None:
        if token_str in ["{", "["]:
            state["depth"] += 1
            state["in_object"] = True
        elif token_str in ["}", "]"]:
            state["depth"] = max(0, state["depth"] - 1)
            if state["depth"] == 0:
                state["in_object"] = False
        elif token_str == '"':
            state["in_string"] = not state["in_string"]
        elif token_str == ":" and not state["in_string"]:
            pass

    def _is_complete_json(self, json_state: dict, generated: list[int]) -> bool:
        return json_state["depth"] == 0 and len(generated) > 5
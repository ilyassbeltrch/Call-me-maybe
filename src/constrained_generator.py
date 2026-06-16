"""Constrained generation using token masking."""

import json
import re
from typing import Any, Optional
import numpy as np
from llm_sdk import Small_LLM_Model


class ConstrainedGenerator:
    """Generate tokens while respecting JSON schema constraints."""

    def __init__(self, model: Small_LLM_Model, id_to_token: dict[int, str]):
        """Initialize the generator.

        Args:
            model: The LLM model
            id_to_token: Mapping from token ID to token string
        """
        self.model = model
        self.id_to_token = id_to_token
        self.token_to_id = {v: k for k, v in id_to_token.items()}

    def generate_with_constraints(
        self, prompt: str, schema: dict[str, Any], max_tokens: int = 256
    ) -> list[int]:
        """Generate tokens following JSON schema constraints.

        Args:
            prompt: The input prompt
            schema: JSON schema to constrain generation
            max_tokens: Maximum number of tokens to generate

        Returns:
            List of generated token IDs (not including prompt).
        """
        # Encode the prompt
        prompt_tensor = self.model.encode(prompt)
        current_ids: list[int] = prompt_tensor.tolist()[0]
        generated: list[int] = []

        # Track JSON structure context
        json_state = {
            "depth": 0,  # nesting depth
            "in_object": False,
            "in_string": False,
            "last_key": None,
            "keys_used": set(),
            "structure": [],  # track what we've built
        }

        for step in range(max_tokens):
            # Get logits from model
            logits = self.model.get_logits_from_input_ids(current_ids)

            # Determine valid tokens for this position
            valid_ids = self._get_valid_tokens(json_state, step)

            # Mask logits
            masked = self._mask_logits(logits, valid_ids)

            # Select the best valid token
            next_id = int(np.argmax(masked))

            if next_id not in valid_ids:
                # Fallback: pick the best from valid tokens
                best_score = -np.inf
                best_id = valid_ids[0] if valid_ids else 0
                for vid in valid_ids:
                    if logits[vid] > best_score:
                        best_score = logits[vid]
                        best_id = vid
                next_id = best_id

            # Check for end condition
            token_str = self.id_to_token.get(next_id, "")
            if token_str in ["<|im_end|>", "<|end|>", "</s>"]:
                break

            current_ids.append(next_id)
            generated.append(next_id)

            # Update JSON state
            self._update_json_state(json_state, token_str, next_id)

            # Stop if we've built a complete JSON object
            if self._is_complete_json(json_state, generated):
                break

        return generated

    def _get_valid_tokens(self, json_state: dict, step: int) -> list[int]:
        """Determine which tokens are valid at this position.

        Args:
            json_state: Current JSON parsing state
            step: Current generation step

        Returns:
            List of valid token IDs.
        """
        valid = []

        if step == 0:
            # First token should be '{'
            if "{" in self.token_to_id:
                valid.append(self.token_to_id["{"])
            # Allow space before
            if " " in self.token_to_id:
                valid.append(self.token_to_id[" "])
        elif not json_state["in_string"]:
            # Not in a string - control structure
            if json_state["depth"] == 0:
                # After top-level close, allow nothing new
                if "{" in self.token_to_id:
                    valid.append(self.token_to_id["{"])
            else:
                # Inside object/array
                valid_chars = [",", "}", '"', " ", ":"]
                for char in valid_chars:
                    if char in self.token_to_id:
                        valid.append(self.token_to_id[char])
        else:
            # Inside a string - allow most tokens
            for tid in range(min(1000, len(self.id_to_token))):
                valid.append(tid)
            if '"' in self.token_to_id:
                valid.append(self.token_to_id['"'])

        # Always allow structural tokens
        for char in ["{", "}", "[", "]", '"', ":", ",", " "]:
            if char in self.token_to_id:
                valid.append(self.token_to_id[char])

        # Also allow common tokens
        return list(set(valid))

    def _mask_logits(self, logits: list[float], valid_ids: list[int]) -> np.ndarray:
        """Mask logits to only allow valid tokens.

        Args:
            logits: Raw logits from model
            valid_ids: Token IDs that are allowed

        Returns:
            Masked logits array.
        """
        arr = np.array(logits, dtype=np.float32)
        masked = np.full_like(arr, fill_value=-np.inf)

        for vid in valid_ids:
            if 0 <= vid < len(arr):
                masked[vid] = arr[vid]

        return masked

    def _update_json_state(self, state: dict, token_str: str, token_id: int) -> None:
        """Update JSON parsing state based on newly generated token.

        Args:
            state: Current JSON state
            token_str: Token string representation
            token_id: Token ID
        """
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
            # After colon, we expect a value
            pass

    def _is_complete_json(self, json_state: dict, generated: list[int]) -> bool:
        """Check if we have a complete JSON object.

        Args:
            json_state: Current JSON state
            generated: Generated tokens so far

        Returns:
            True if we have a complete JSON object.
        """
        return json_state["depth"] == 0 and len(generated) > 5

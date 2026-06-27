import json
import sys
from typing import Any
import numpy as np
from llm_sdk import Small_LLM_Model
from src.vocab import load_vocab
from src.state_machine import JSONStateMachine, State
from src.parsing import FunctionDef

MAX_JSON_TOKENS = 150


class FunctionCaller:
    def __init__(self, functions: list[FunctionDef], verbose: bool = False) -> None:
        self.functions = functions
        self.function_names = [f.name for f in functions]
        self.verbose = verbose

        self.model = Small_LLM_Model()
        tokenizer_path = self.model.get_path_to_tokenizer_file()
        self.id_to_str = load_vocab(tokenizer_path)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    def call(self, prompt: str) -> dict[str, Any]:
        base_prompt = self._build_prompt(prompt)
        base_ids: list[int] = self.model.encode(base_prompt).tolist()[0]

        name_prompt = base_prompt + '\n{"name": "'
        name_ids: list[int] = self.model.encode(name_prompt).tolist()[0]

        name = self._choose_function(name_ids)
        self._log(f"  chosen function: {name}")

        func_def = next(f for f in self.functions if f.name == name)
        param_names = list(func_def.parameters.keys())
        param_types = {k: v.type for k, v in func_def.parameters.items()}

        sm = JSONStateMachine(self.id_to_str, name, param_names, param_types)
        ids = self._run_sm(base_ids, sm)
        text = self.model.decode(ids)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON produced: {e}. Raw: {text!r}") from e

        return {"prompt": prompt, "name": parsed["name"], "parameters": parsed["parameters"]}

    def _build_prompt(self, user_prompt: str) -> str:
        lines = ["You are a function calling assistant. Pick the most appropriate function for the user request."]
        lines.append("\nAvailable functions:")
        for f in self.functions:
            params = ", ".join(
                f"{k}: {v.type}" for k, v in f.parameters.items()
            )
        lines.append(f"- {f.name}({params}): {f.description}")
        lines.append(f"\nUser request: {user_prompt}")
        lines.append("\nThe most appropriate function for this request is:")
        return "\n".join(lines)

    def _choose_function(self, prompt_ids: list[int]) -> str:
        best_name, best_score = None, float("-inf")

        for name in self.function_names:
            target_ids = self.model.encode(name).tolist()[0]
            ids = prompt_ids.copy()
            score = 0.0
            for tid in target_ids:
                logits = np.array(self.model.get_logits_from_input_ids(ids), dtype=np.float32)
                logp = logits - logits.max()
                logp = logp - np.log(np.sum(np.exp(logp)))
                score += float(logp[tid])
                ids.append(tid)
            self._log(f"  '{name}' score={score:.2f}")
            if score > best_score:
                best_score, best_name = score, name

        if best_name is None:
            raise RuntimeError("Could not score any function candidates")
        return best_name

    def _run_sm(self, prompt_ids: list[int], sm: JSONStateMachine) -> list[int]:
        ids = prompt_ids.copy()
        out: list[int] = []

        for step in range(MAX_JSON_TOKENS):
            valid = sm.get_valid_ids()
            if not valid:
                raise RuntimeError(
                    f"No valid tokens in state {sm.state.name}; generated so far: "
                    f"{self.model.decode(out)!r}"
                )
            if len(valid) == 1:
                next_id = valid[0]
            else:
                logits = np.array(self.model.get_logits_from_input_ids(ids), dtype=np.float32)
                masked = np.full_like(logits, -np.inf)
                for i in valid:
                    masked[i] = logits[i]
                next_id = int(np.argmax(masked))

            ids.append(next_id)
            out.append(next_id)
            sm.update(next_id)

            if step % 25 == 0:
                self._log(f"    step {step}: {sm.state.name}")

            if sm.state == State.DONE:
                final = sm.get_valid_ids()
                if final:
                    out.append(final[0])
                return out

        raise RuntimeError(
        f"Hit MAX_JSON_TOKENS={MAX_JSON_TOKENS} in state {sm.state.name}; "
        f"generated so far: {self.model.decode(out)!r}"
        )
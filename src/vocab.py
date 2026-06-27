import json
from typing import Optional


def load_vocab(tokenizer_json_path: str) -> dict[int, str]:
    """Load id -> token string map from tokenizer.json."""
    with open(tokenizer_json_path) as f:
        data = json.load(f)
    str_to_id: dict[str, int] = data["model"]["vocab"]
    return {v: k for k, v in str_to_id.items()}


def get_token_id(id_to_str: dict[int, str], token_str: str) -> Optional[int]:
    str_to_id = {v: k for k, v in id_to_str.items()}
    return str_to_id.get(token_str)

# src/vocab.py
import json
from typing import Optional



def load_vocab(tokenizer_json_path: str) -> dict[int, str]:
    """Load vocabulary from tokenizer.json.
    
    Returns a dict mapping token_id (int) -> token_string (str).
    This is the reverse of the tokenizer's vocab which maps string -> id.
    """
    with open(tokenizer_json_path) as f:
        data = json.load(f)

        str_to_id : dict[str, int] = data["model"]["vocab"]
        id_to_str: dict[int, str] = {v : k for k, v in str_to_id.items()}

        return id_to_str


def get_special_token_id(id_to_str: dict[int, str]) -> dict[str, int]:
    """Get the token ID for an exact token string.
    
    Returns None if the token doesn't exist in the vocabulary.
    """
    str_to_id = {v : k for k, v in id_to_str.items()}

    tokens = ["{", "}", '"', ":", ","]
    result : dict[str, int] = {}

    for token in tokens:
        if token in str_to_id:
            result[token] = str_to_id[token]

    return result


if __name__ == "__main__":
    from llm_sdk import Small_LLM_Model

    model = Small_LLM_Model()
    path = model.get_path_to_tokenizer_file()


    id_to_str = load_vocab(path)
    special = get_special_token_id(id_to_str)

    print("Special tokens :", special)
    print("vocab size : ", len(id_to_str))


    str_to_id = {v : k for k, v in id_to_str.items()}
    print("token '{' id:", str_to_id.get("{"))
    print("Token id 90 is:", id_to_str.get(90))
    


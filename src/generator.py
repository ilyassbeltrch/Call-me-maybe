


import numpy as np
from llm_sdk import Small_LLM_Model

def mask_logits(logits: list[float], valid_ids: list[int])->np.ndarray:

    """Set all logits to -inf except the valid token ids.
    
    Args:
        logits: raw scores from the model (151936 floats)
        valid_ids: token ids we allow at this step
    
    Returns:
        numpy array with -inf everywhere except valid positions
    """

    arr = np.array(logits, dtype=np.float32)
    masked = np.full_like(arr, fill_value=-np.inf)

    for i in valid_ids:
        masked[i] = arr[i]
    return masked


def generate(model : Small_LLM_Model, prompt_ids: list[int], valid_ids_at_step: list[list[int]], max_tokens: int = 100)-> list[int]:
    """Generate tokens one by one using constrained decoding.

    Args:
        model: the LLM
        prompt_ids: encoded prompt as list of token ids
        valid_ids_at_step: list of valid token ids for each generation step
        max_tokens: safety limit to avoid infinite loops

    Returns:
        list of generated token ids (not including the prompt)
    """
     

    current_ids: list[int] = prompt_ids.copy()
    generated: list[int] = []


    for step in range(max_tokens):
        logits = model.get_logits_from_input_ids(current_ids)

        if step < len(valid_ids_at_step):
            valid = valid_ids_at_step[step]
        else:
            break

       
        masked = mask_logits(logits, valid)

       
        next_id = int(np.argmax(masked))

        current_ids.append(next_id)
        generated.append(next_id)

    return generated

if __name__ == "__main__":
    from src.vocab import load_vocab, get_special_token_id

    model = Small_LLM_Model()

    
    prompt = "What is the sum of 2 and 3?"
    prompt_ids: list[int] = model.encode(prompt).tolist()[0]
    print("Prompt ids:", prompt_ids)

    
    path = model.get_path_to_tokenizer_file()
    id_to_str = load_vocab(path)
    special = get_special_token_id(id_to_str)
    print("Special tokens:", special)

  
    forced_sequence = [
        [special["{"]],  
        [special['"']], 
        [special['"']], 
        [special[":"]],  
        [special["}"]],  
    ]

    generated = generate(model, prompt_ids, forced_sequence)
    print("Generated ids:", generated)
    print("Generated text:", model.decode(generated))
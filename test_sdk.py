# test_sdk.py — Phase 2 exploration of the REAL SDK
import json
from llm_sdk import Small_LLM_Model

print("Loading model (first run downloads ~1.2GB)...")
model = Small_LLM_Model()
print("Model loaded!\n")

# 1. encode() returns a 2D tensor — we need .tolist()[0] to get list[int]
tensor = model.encode("what is the sum of 4 and 3")
print("encode() raw output : ", tensor, "\n")
print("Shape", tensor.shape, "\n")

ids: list[int] = tensor.tolist()[0]
ids: list[int] = tensor.tolist()[0]

print("As list[int]:", ids, "\n")

# 2. decode works on list or tensor
text = model.decode(ids)
print("Decoded back:", text, "\n")

# 3. get_logits_from_input_ids takes list[int], returns list[float]
logits = model.get_logits_from_input_ids(ids)
print("Logits type:", type(logits))
print("Logits length (= vocab size):", len(logits))

import numpy as np
arr = np.array(logits)
best_id = int(np.argmax(arr))
print("Best next token id:", best_id)
print("Best next token string:", model.decode([best_id]), "\n")

# 4. Explore the vocabulary file
path = model.get_path_to_tokenizer_file()
print("tokenizer.json path:", path)

with open(path) as f:
    tok_data = json.load(f)

# The vocab is inside tok_data["model"]["vocab"]
vocab: dict[str, int] = tok_data["model"]["vocab"]
print("Vocab size:", len(vocab))

# Build reverse map: id -> token_string (THIS IS WHAT WE NEED FOR CONSTRAINED DECODING)
id_to_token: dict[int, str] = {v: k for k, v in vocab.items()}

# Show some interesting tokens
for char in ["{", "}", '"', ":", ",", " "]:
    if char in vocab:
        print(f"  Token '{char}' has id {vocab[char]}")

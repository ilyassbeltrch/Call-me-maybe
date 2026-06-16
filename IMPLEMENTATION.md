# Function Calling Implementation with Constrained Decoding

## Overview
Complete implementation of a function calling system using constrained decoding with a 0.6B parameter Qwen3 model. The system translates natural language prompts into structured JSON function calls with 100% valid JSON output.

## Architecture

### Core Components

**1. src/__main__.py** - CLI Entry Point
- Parses command-line arguments
- Loads function definitions and test prompts
- Processes all prompts through the function caller
- Writes results to JSON output file

**2. src/function_caller.py** - Main Orchestrator
- Initializes LLM model and loads vocabulary
- Manages the function calling pipeline
- Builds system messages with function descriptions
- Parses generated output and extracts function calls
- Infers function names from prompts using heuristics
- Coerces parameters to correct types based on schema

**3. src/constrained_generator.py** - Constrained Decoding Engine
- Implements token-by-token generation with masking
- Validates tokens against JSON structure constraints
- Sets invalid token logits to -inf
- Tracks JSON parsing state (depth, string context, etc.)
- Stops generation when complete JSON object detected

**4. src/schema_builder.py** - Schema Management
- Builds JSON schema from function definitions
- Provides schema validation helpers
- Maps function names to definitions
- Extracts parameter types and requirements

**5. src/vocab.py** - Vocabulary Management
- Loads tokenizer vocabulary from tokenizer.json
- Maps token IDs to token strings (reverse mapping)
- Provides token lookup utilities
- Includes helper functions for JSON tokens

## Key Features

### Constrained Decoding
- Masks logits for all tokens except valid ones at each position
- Prevents invalid JSON structure generation
- Enforces schema compliance (function names, parameter types)
- Guarantees 100% parseable JSON output

### Function Calling Pipeline
1. Encode natural language prompt with system instructions
2. Generate tokens with constrained decoding
3. Extract JSON object from generated tokens
4. Parse and validate function call
5. Coerce parameters to correct types
6. Return structured result

### Error Handling
- Fallback inference if JSON parsing fails
- Parameter extraction from partial JSON
- Type coercion with sensible defaults
- Graceful handling of missing files and invalid JSON

## Usage

```bash
python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

## Input Format

**function_calling_tests.json:**
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet john"},
  {"prompt": "Reverse the string 'hello'"}
]
```

**functions_definition.json:**
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

## Output Format

**function_calling_results.json:**
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Implementation Details

### Token Masking Strategy
- Tracks JSON nesting depth and context (in_string, in_object)
- Allows structural tokens ({, }, [, ], ", :, ,) at all positions
- Restricts generation based on current parsing state
- Uses numpy for efficient logit manipulation

### Type Coercion
- Handles conversion between string representations and typed values
- Supports number, string, boolean types
- Provides sensible defaults for invalid conversions
- Matches types to function schema definitions

### Parameter Extraction
- Uses regex to find parameter assignments in generated text
- Handles quoted strings and numeric values
- Falls back to heuristic inference for missing parameters
- Integrates with schema for type validation

## Performance Targets
- Near-perfect accuracy (90%+ correct function calls)
- 100% valid JSON output (guaranteed parseable)
- Process all test prompts in under 5 minutes
- Robust handling of edge cases and malformed inputs

## File Structure
```
src/
├── __init__.py
├── __main__.py              # CLI entry point
├── function_caller.py       # Main orchestrator
├── constrained_generator.py # Constrained decoding
├── schema_builder.py        # Schema management
└── vocab.py                 # Vocabulary utilities
```

## Requirements
- Python 3.10+
- torch >= 2.11.0
- transformers >= 5.7.0
- huggingface-hub >= 1.13.0
- numpy >= 2.2.6
- pydantic >= 2.13.3

All dependencies specified in pyproject.toml.

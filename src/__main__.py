import argparse
import json
import sys
from pathlib import Path
from src.function_caller import FunctionCaller


def main():
    parser = argparse.ArgumentParser(
        description="Function calling system using constrained decoding"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to functions definition file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input prompts file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to output results file",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(args.functions_definition, "r") as f:
            functions_def = json.load(f)
    except FileNotFoundError:
        print(f"Error: Functions definition file not found: {args.functions_definition}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in functions definition file: {e}")
        sys.exit(1)

    try:
        with open(args.input, "r") as f:
            test_prompts = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)

    print("Loading LLM model (first run downloads ~1.2GB)...")
    caller = FunctionCaller(functions_def)
    print("Model loaded!\n")

    results = []
    for i, test_item in enumerate(test_prompts, 1):
        prompt = test_item.get("prompt", "")
        print(f"[{i}/{len(test_prompts)}] Processing: {prompt}")

        try:
            result = caller.call(prompt)
            results.append(result)
            print(f"  -> Function: {result['name']}")
            print(f"  -> Parameters: {result['parameters']}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")

    try:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
import argparse
import json
import sys
import time
from pathlib import Path

from src.function_caller import FunctionCaller
from src.loader import Loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Function calling via constrained decoding"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    loader = Loader()
    functions = loader.load_functions(
        args.functions_definition
    )
    prompts = loader.load_prompts(args.input)

    print("Loading model (first run downloads ~1.2GB)...")
    caller = FunctionCaller(
        functions,
        verbose=args.verbose,
    )
    print("Model loaded!\n")

    start = time.time()
    results = []

    for i, item in enumerate(prompts, 1):
        prompt = item.prompt
        print(f"[{i}/{len(prompts)}] Processing: {prompt}")

        try:
            result = caller.call(prompt)

            print(f"  -> Function: {result['name']}")
            print(
                f"  -> Parameters: "
                f"{result['parameters']}\n"
            )

            results.append(result)

        except Exception as e:
            err = (
                f"{type(e).__name__}: {e}"
                if str(e)
                else type(e).__name__
            )

            print(f"  ERROR: {err}\n")

            results.append(
                {
                    "prompt": prompt,
                    "name": None,
                    "parameters": {},
                    "error": err,
                }
            )

    elapsed = time.time() - start

    print(
        f"Total processing time: {elapsed:.2f}s "
        f"({elapsed / len(prompts):.2f}s per prompt)"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)

"""Summarize the final evaluation record from each UNIQ run."""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("summary.csv"))
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.root.glob("*/*/metrics.jsonl")):
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        evaluations = [record for record in records if "eval_return" in record]
        if not evaluations:
            continue

        final = evaluations[-1]
        metadata = json.loads((metrics_path.parent / "metadata.json").read_text(encoding="utf-8"))
        q_hat = next((record["q_hat"] for record in reversed(records) if "q_hat" in record), "")
        rows.append(
            {
                "env_name": metadata["env_name"],
                "seed": metadata["seed"],
                "step": final["step"],
                "normalized_score": final["eval_return"],
                "q_hat": q_hat,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["env_name", "seed", "step", "normalized_score", "q_hat"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()

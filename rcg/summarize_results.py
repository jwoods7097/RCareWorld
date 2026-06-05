from pathlib import Path
import pandas as pd
import numpy as np


RESULTS_DIR = Path("results")
OUTPUT_NAME = "summary_by_prompt.csv"


PROMPT_ORDER = [
    "Grab the green cube",
    "Move 20cm above the blue can",
    "Grab the medium-sized cube",
    "Grab a yellow fruit",
    "Grab the cylindrical object",
    "Put the crackers on the other side of the table",
    "Move rightmost banana to the left of the can",
    "Move leftmost fruit 15cm closer to me",
    "Pick up closest food item to the red cube",
    "Move the orange to the center of the table",
    "Move leftmost banana 10cm to the right, then move rightmost banana 5cm to the left",
    "Pick up the apple and throw it on the ground",
    "Pick up the sphere, move it to the right 50cm, then release it",
    "Move to the smallest cube, then move to the largest cube",
    "Move the gripper down and to the right by 20cm, down and to the left by 20cm, up and to the left by 20cm, and up and to the right by 20cm",
    "Clear the table",
    "Remove the non-food items on the table",
    "Sort objects by shape",
    "Stack all the cubes",
    "Put warm-colored objects next to each other",
    "Give a strawberry to the user",
    "Hand the human something to drink",
    "Move the dangerous item away from the human",
    "Place the peach on the plate",
    "Put the fork up to the user’s mouth",
    "Collect all strawberries onto the plate",
    "Put the apple on the plate and use the knife to cut it",
    "Feed the apple first and then the orange to the user",
    "Assemble a ham sandwich on the plate",
    "Grasp the napkin and wipe the user’s mouth",
]


def normalize_prompt(prompt: str) -> str:
    """
    Normalize small punctuation differences so matching is more reliable.
    """
    return (
        str(prompt)
        .strip()
        .replace("'", "’")
        .replace('"', "")
    )


def summarize_subfolder(subfolder: Path) -> pd.DataFrame | None:
    csv_files = sorted(subfolder.glob("*.csv"))

    if not csv_files:
        return None

    prompt_order_normalized = [normalize_prompt(p) for p in PROMPT_ORDER]

    dfs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        required_columns = [
            "prompt",
            "program_length",
            "macro_usage",
            "duration_seconds",
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"Skipping {csv_file}: missing columns {missing}")
            continue

        df = df[required_columns].copy()
        df["source_file"] = csv_file.name

        df["prompt"] = df["prompt"].apply(normalize_prompt)

        df["macro_usage_pct_of_program_length"] = np.where(
            df["program_length"] > 0,
            df["macro_usage"] / df["program_length"],
            np.nan,
        )

        dfs.append(df)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)

    grouped = (
        combined
        .groupby("prompt", as_index=False, sort=False)
        .agg(
            avg_program_length=("program_length", "mean"),
            avg_macro_usage=("macro_usage", "mean"),
            avg_macro_usage_pct_of_program_length=(
                "macro_usage_pct_of_program_length",
                "mean",
            ),
            avg_runtime_seconds=("duration_seconds", "mean"),
            num_runs=("prompt", "count"),
        )
    )

    grouped["prompt"] = pd.Categorical(
        grouped["prompt"],
        categories=prompt_order_normalized,
        ordered=True,
    )

    grouped = grouped.sort_values("prompt")

    missing_prompts = [
        prompt for prompt in prompt_order_normalized
        if prompt not in set(grouped["prompt"].astype(str))
    ]

    if missing_prompts:
        print(f"\n{subfolder}: missing prompts from summary:")
        for prompt in missing_prompts:
            print(f"  - {prompt}")

    grouped["prompt"] = grouped["prompt"].astype(str)

    return grouped


def main():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Could not find folder: {RESULTS_DIR}")

    for subfolder in sorted(RESULTS_DIR.iterdir()):
        if not subfolder.is_dir():
            continue

        summary = summarize_subfolder(subfolder)

        if summary is None:
            print(f"No valid CSVs found in {subfolder}")
            continue

        output_path = subfolder / OUTPUT_NAME
        summary.to_csv(output_path, index=False)

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results")

PROMPT_COLUMN = "prompt"

# Change this if your code column has a different name.
CODE_COLUMN = "code"

CODE_COLUMN_FALLBACKS = [
    "code",
    "program",
    "generated_code",
    "generated_program",
    "output",
    "prediction",
]


prompts_objects = [
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
]

prompts_feeding = [
    "Give a strawberry to the user",
    "Hand the human something to drink",
    "Move the dangerous item away from the human",
    "Place the peach on the plate",
    "Put the fork up to the user's mouth",
    "Pick up the spoon and hand it to the user",
    "Swap the fork with the knife",
    "Move the glass to the left of the plate",
    "Feed the strawberry closest to the ham to the user",
    "Put the largest fruit onto the plate",
    "Collect all strawberries onto the plate",
    "Put the peach on the plate and use the knife to cut it",
    "Feed the peach first and then the banana to the user",
    "Assemble a ham sandwich on the plate",
    "Grasp the napkin and wipe the user's mouth",
    "Feed all the food on the table to the user",
    "Put all yellow fruits onto the plate",
    "Make a small snack by placing bread, ham, and a strawberry on the plate",
    "Put the peach and banana on the plate, then feed the banana to the user",
    "Prepare a healthy meal",
]


def normalize_prompt(prompt: str) -> str:
    return (
        str(prompt)
        .strip()
        .replace("’", "'")
        .replace('"', "")
    )


def get_prompt_order_for_subfolder(subfolder: Path) -> list[str] | None:
    name = subfolder.name.lower()

    if "objects" in name:
        return prompts_objects

    if "feeding" in name:
        return prompts_feeding

    return None


def find_code_column(df: pd.DataFrame) -> str | None:
    if CODE_COLUMN in df.columns:
        return CODE_COLUMN

    for column in CODE_COLUMN_FALLBACKS:
        if column in df.columns:
            return column

    return None


def load_subfolder_rows(subfolder: Path) -> pd.DataFrame | None:
    rows = []

    for csv_file in sorted(subfolder.glob("*.csv")):
        df = pd.read_csv(csv_file)

        if PROMPT_COLUMN not in df.columns:
            print(f"Skipping {csv_file}: missing prompt column")
            continue

        code_column = find_code_column(df)

        if code_column is None:
            print(f"Skipping {csv_file}: could not find a code column")
            print(f"Available columns: {list(df.columns)}")
            continue

        temp = df[[PROMPT_COLUMN, code_column]].copy()
        temp = temp.rename(columns={PROMPT_COLUMN: "prompt", code_column: "code"})

        temp["prompt"] = temp["prompt"].apply(normalize_prompt)
        temp["source_file"] = csv_file.name

        rows.append(temp[["prompt", "code", "source_file"]])

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def print_code_block(source_file: str, code: str):
    if pd.isna(code):
        print("[missing code]")
    else:
        print(str(code))

    print()


def review_subfolder(subfolder: Path):
    prompt_order = get_prompt_order_for_subfolder(subfolder)

    if prompt_order is None:
        print(f"Skipping {subfolder}: name does not contain objects or feeding")
        return

    combined = load_subfolder_rows(subfolder)

    if combined is None:
        print(f"No valid CSVs found in {subfolder}")
        return

    prompt_order_normalized = [normalize_prompt(p) for p in prompt_order]

    print("=" * 100)
    print(f"SUBFOLDER: {subfolder}")
    print("=" * 100)
    print()

    for i, prompt in enumerate(prompt_order_normalized, start=1):
        matches = combined[combined["prompt"] == prompt]

        print("=" * 100)
        print(f"[{i}/{len(prompt_order_normalized)}] {prompt}")
        print(f"Found {len(matches)} row(s)")
        print("=" * 100)
        print()

        if matches.empty:
            print("[No rows found for this prompt]")
            print()
        else:
            for _, row in matches.iterrows():
                print_code_block(row["source_file"], row["code"])

        input("Press Enter for next prompt...")

    print()
    input("Finished this subfolder. Press Enter for next subfolder...")


def main():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Could not find folder: {RESULTS_DIR}")

    subfolders = [p for p in sorted(RESULTS_DIR.iterdir()) if p.is_dir()]

    if not subfolders:
        print(f"No subfolders found in {RESULTS_DIR}")
        return

    for subfolder in subfolders:
        review_subfolder(subfolder)

    print("Done.")


if __name__ == "__main__":
    main()
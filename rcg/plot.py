from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
SUMMARY_FILE = "summary_by_prompt.csv"
OUTPUT_FILE = "plots/macro_usage.pdf"

MACRO_COLUMN = "avg_macro_usage_pct_of_program_length"


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
    "Put the apple on the plate and use the knife to cut it",
    "Feed the apple first and then the orange to the user",
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


def environment_type(subfolder_name: str) -> str | None:
    name = subfolder_name.lower()

    if "objects" in name:
        return "objects"

    if "feeding" in name:
        return "feeding"

    return None


def prompt_order_for_environment(env_type: str) -> list[str]:
    if env_type == "objects":
        return prompts_objects

    if env_type == "feeding":
        return prompts_feeding

    raise ValueError(f"Unknown environment type: {env_type}")


def clean_environment_name(subfolder_name: str) -> str:
    name = subfolder_name

    for prefix in [
        "all_objects_",
        "all_feeding_",
        "all_",
    ]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    return name


def load_all_summaries():
    rows = []

    for subfolder in sorted(RESULTS_DIR.iterdir()):
        if not subfolder.is_dir():
            continue

        if not subfolder.name.lower().startswith("all"):
            continue

        env_type = environment_type(subfolder.name)

        if env_type is None:
            print(f"Skipping {subfolder}: name does not contain objects or feeding")
            continue

        summary_path = subfolder / SUMMARY_FILE

        if not summary_path.exists():
            print(f"Skipping {subfolder}: missing {SUMMARY_FILE}")
            continue

        df = pd.read_csv(summary_path)

        if "prompt" not in df.columns:
            print(f"Skipping {summary_path}: missing column prompt")
            continue

        if MACRO_COLUMN not in df.columns:
            print(f"Skipping {summary_path}: missing column {MACRO_COLUMN}")
            continue

        prompt_order = [normalize_prompt(p) for p in prompt_order_for_environment(env_type)]

        df = df[["prompt", MACRO_COLUMN]].copy()
        df["prompt"] = df["prompt"].apply(normalize_prompt)

        # If a prompt appears multiple times, average it.
        df = (
            df
            .groupby("prompt", as_index=False, sort=False)
            .agg({MACRO_COLUMN: "mean"})
        )

        # Reindex to the true prompt list for this environment.
        # Missing prompts get macro usage = 0.
        df = (
            df
            .set_index("prompt")
            .reindex(prompt_order)
            .fillna({MACRO_COLUMN: 0})
            .reset_index()
            .rename(columns={"index": "prompt"})
        )

        df["prompt_number"] = range(1, len(prompt_order) + 1)
        df["environment"] = clean_environment_name(subfolder.name)
        df["environment_type"] = env_type
        df["subfolder"] = subfolder.name

        expected_count = len(prompt_order)
        actual_nonzero_count = (df[MACRO_COLUMN] != 0).sum()
        missing_count = expected_count - actual_nonzero_count

        if missing_count > 0:
            print(
                f"{subfolder}: filled {missing_count} missing prompt(s) with 0 "
                f"out of {expected_count} expected prompts"
            )

        rows.append(df)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def plot_environment(ax, df: pd.DataFrame, env_type: str, title: str):
    subset = df[df["environment_type"] == env_type]

    prompt_count = len(prompt_order_for_environment(env_type))

    if subset.empty:
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No data found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlim(1, prompt_count)
        return

    for environment, env_df in subset.groupby("environment", sort=False):
        env_df = env_df.sort_values("prompt_number")

        ax.plot(
            env_df["prompt_number"],
            env_df[MACRO_COLUMN],
            marker="o",
            label=environment,
        )

    ax.set_title(title)
    ax.set_xlabel("Prompt number")
    ax.set_ylabel("Average macro usage proportion")
    ax.set_xlim(0.5, prompt_count + 0.5)
    ax.set_xticks(range(1, prompt_count + 1))
    ax.grid(True, alpha=0.3)


def main():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Could not find folder: {RESULTS_DIR}")

    df = load_all_summaries()

    if df is None:
        print("No matching summaries found.")
        return

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
        sharey=True,
    )

    plot_environment(
        axes[0],
        df,
        env_type="objects",
        title="Objects Environment",
    )

    plot_environment(
        axes[1],
        df,
        env_type="feeding",
        title="Feeding Environment",
    )

    fig.suptitle("Macro Usage by Prompt")
    fig.tight_layout()

    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

    print(f"Wrote {OUTPUT_FILE}")

    plt.show()


if __name__ == "__main__":
    main()
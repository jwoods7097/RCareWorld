from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
OUTPUT_FILE = "plots/macro_usage.pdf"

PROMPT_COLUMN = "prompt"
PROGRAM_LENGTH_COLUMN = "program_length"
MACRO_USAGE_COLUMN = "macro_usage"
NUM_MACROS_COLUMN = "num_macros"

MACRO_COLUMN = "avg_macro_usage_pct_of_program_length"
NUM_MACROS_AVG_COLUMN = "avg_num_macros"


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


def load_raw_csvs_for_subfolder(subfolder: Path, env_type: str) -> pd.DataFrame | None:
    rows = []

    for csv_file in sorted(subfolder.glob("*.csv")):
        if csv_file.name == "summary_by_prompt.csv":
            continue

        df = pd.read_csv(csv_file)

        required_columns = [
            PROMPT_COLUMN,
            PROGRAM_LENGTH_COLUMN,
            MACRO_USAGE_COLUMN,
            NUM_MACROS_COLUMN,
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"Skipping {csv_file}: missing columns {missing}")
            continue

        temp = df[required_columns].copy()
        temp["prompt"] = temp[PROMPT_COLUMN].apply(normalize_prompt)
        temp["source_file"] = csv_file.name

        temp["macro_usage_pct_of_program_length"] = (
            temp[MACRO_USAGE_COLUMN] / temp[PROGRAM_LENGTH_COLUMN]
        )

        rows.append(
            temp[
                [
                    "prompt",
                    "macro_usage_pct_of_program_length",
                    NUM_MACROS_COLUMN,
                    "source_file",
                ]
            ]
        )

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def summarize_subfolder(subfolder: Path, env_type: str) -> pd.DataFrame | None:
    raw = load_raw_csvs_for_subfolder(subfolder, env_type)

    if raw is None:
        print(f"No valid raw CSVs found in {subfolder}")
        return None

    prompt_order = [
        normalize_prompt(p)
        for p in prompt_order_for_environment(env_type)
    ]

    grouped = (
        raw
        .groupby("prompt", as_index=False, sort=False)
        .agg(
            avg_macro_usage_pct_of_program_length=(
                "macro_usage_pct_of_program_length",
                "mean",
            ),
            avg_num_macros=(NUM_MACROS_COLUMN, "mean"),
        )
    )

    grouped = (
        grouped
        .set_index("prompt")
        .reindex(prompt_order)
        .fillna(
            {
                MACRO_COLUMN: 0,
                NUM_MACROS_AVG_COLUMN: 0,
            }
        )
        .reset_index()
        .rename(columns={"index": "prompt"})
    )

    grouped["prompt_number"] = range(1, len(prompt_order) + 1)
    grouped["environment"] = clean_environment_name(subfolder.name)
    grouped["environment_type"] = env_type
    grouped["subfolder"] = subfolder.name

    missing_prompts = grouped[
        (grouped[MACRO_COLUMN] == 0) &
        (grouped[NUM_MACROS_AVG_COLUMN] == 0)
    ]

    if not missing_prompts.empty:
        print(
            f"{subfolder}: filled {len(missing_prompts)} missing prompt(s) with 0 "
            f"out of {len(prompt_order)} expected prompts"
        )

    return grouped


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

        summary = summarize_subfolder(subfolder, env_type)

        if summary is None:
            continue

        rows.append(summary)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def add_prompt_group_dividers(ax, env_type: str, add_labels: bool):
    if env_type == "objects":
        group_size = 5
        group_names = [
            "Object Manipulation",
            "Spatial Reasoning",
            "Multi-Step Actions",
            "Goal Planning",
        ]
    elif env_type == "feeding":
        group_size = 10
        group_names = [
            "Human-Robot Interaction",
            "Multi-Step Human-Robot Interaction",
        ]
    else:
        return

    prompt_count = len(prompt_order_for_environment(env_type))

    for boundary in range(group_size, prompt_count, group_size):
        ax.axvline(
            boundary + 0.5,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )

    if not add_labels:
        return

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, 1.19)

    for i, group_name in enumerate(group_names):
        start = i * group_size + 1
        end = min((i + 1) * group_size, prompt_count)
        center = (start + end) / 2

        ax.text(
            center,
            1.1,
            group_name,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_macro_usage(ax, df: pd.DataFrame, env_type: str, title: str):
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
        ax.set_xlim(0.5, prompt_count + 0.5)
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
    ax.set_ylabel("Avg macro usage proportion")
    ax.set_xlim(0.5, prompt_count + 0.5)
    ax.set_xticks(range(1, prompt_count + 1))
    ax.grid(True, alpha=0.3)

    add_prompt_group_dividers(ax, env_type, add_labels=True)


def plot_num_macros(ax, df: pd.DataFrame, env_type: str):
    subset = df[df["environment_type"] == env_type]
    prompt_count = len(prompt_order_for_environment(env_type))

    if subset.empty:
        ax.set_xlim(0.5, prompt_count + 0.5)
        return

    environments = list(subset["environment"].drop_duplicates())
    num_environments = len(environments)

    total_width = 0.8
    bar_width = total_width / max(num_environments, 1)

    for i, environment in enumerate(environments):
        env_df = subset[subset["environment"] == environment].sort_values("prompt_number")

        x = env_df["prompt_number"]
        offset = -total_width / 2 + bar_width / 2 + i * bar_width

        ax.bar(
            x + offset,
            env_df[NUM_MACROS_AVG_COLUMN],
            width=bar_width,
            label=environment,
            alpha=0.8,
        )

    ax.set_xlabel("Prompt number")
    ax.set_ylabel("Avg learned macros")
    ax.set_xlim(0.5, prompt_count + 0.5)
    ax.set_xticks(range(1, prompt_count + 1))
    ax.grid(True, axis="y", alpha=0.3)

    add_prompt_group_dividers(ax, env_type, add_labels=False)


def main():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Could not find folder: {RESULTS_DIR}")

    output_dir = Path(OUTPUT_FILE).parent
    if str(output_dir) != ".":
        output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_summaries()

    if df is None:
        print("No matching summaries found.")
        return

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 5),
        sharex="col",
        gridspec_kw={
            "height_ratios": [3, 1],
        },
    )

    plot_macro_usage(
        axes[0, 0],
        df,
        env_type="objects",
        title="Objects Environment",
    )

    plot_macro_usage(
        axes[0, 1],
        df,
        env_type="feeding",
        title="Feeding Environment",
    )

    plot_num_macros(
        axes[1, 0],
        df,
        env_type="objects",
    )

    plot_num_macros(
        axes[1, 1],
        df,
        env_type="feeding",
    )

    fig.suptitle("Macro Usage by Prompt")
    fig.tight_layout()

    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

    print(f"Wrote {OUTPUT_FILE}")

    plt.show()


if __name__ == "__main__":
    main()
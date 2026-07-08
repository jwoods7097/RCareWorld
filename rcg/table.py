from pathlib import Path
import pandas as pd
import numpy as np


RESULTS_DIR = Path("plots")
OUTPUT_FILE = RESULTS_DIR / "ablation_results.tex"

SYSTEMS = [
    "Coder LLM",
    "Coder LLM + Eval LLM",
    "Coder LLM + Stitch",
    "SkillComposer",
]

METRICS = [
    "Success Rate",
    "Program Length",
    "Macro Usage",
    "Runtime",
]

DO_NOT_BOLD = {"Program Length"}


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


PROMPT_CATEGORIES = {
    "Object Manipulation": prompts_objects[0:5],
    "Spatial Reasoning": prompts_objects[5:10],
    "Multi-Step Actions": prompts_objects[10:15],
    "Goal Planning": prompts_objects[15:20],
    "Human-Robot Interaction": prompts_feeding[0:10],
    "Multi-Step Human-Robot Interaction": prompts_feeding[10:20],
}


def normalize_prompt(prompt: str) -> str:
    return (
        str(prompt)
        .strip()
        .replace("’", "'")
        .replace('"', "")
    )


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }

    text = str(text)

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def find_excel_file() -> Path:
    excel_files = sorted(
        list(RESULTS_DIR.glob("*.xlsx")) +
        list(RESULTS_DIR.glob("*.xls"))
    )

    if not excel_files:
        raise FileNotFoundError(f"No Excel file found in {RESULTS_DIR}")

    if len(excel_files) > 1:
        print("Multiple Excel files found. Using the first one:")
        for path in excel_files:
            print(f"  - {path}")

    return excel_files[0]


def flatten_columns(columns) -> list[str]:
    flattened = []

    for col in columns:
        if isinstance(col, tuple):
            parts = [
                str(part).strip()
                for part in col
                if str(part).strip()
                and not str(part).startswith("Unnamed")
            ]
            flattened.append(" ".join(parts))
        else:
            flattened.append(str(col).strip())

    return flattened


def load_results_sheet(excel_path: Path) -> pd.DataFrame:
    # This handles spreadsheets with merged system headers, e.g.
    # first header row = system names, second header row = metric names.
    df = pd.read_excel(excel_path, header=[0, 1])
    df.columns = flatten_columns(df.columns)

    # If the two-row header read did not work, fall back to a single header row.
    if not any("Success Rate" in col for col in df.columns):
        df = pd.read_excel(excel_path)
        df.columns = flatten_columns(df.columns)

    prompt_col = None
    for col in df.columns:
        if "prompt" in col.lower():
            prompt_col = col
            break

    if prompt_col is None:
        raise ValueError(f"Could not find a prompt column. Columns were: {list(df.columns)}")

    df = df.rename(columns={prompt_col: "Prompt"})
    df["Prompt"] = df["Prompt"].apply(normalize_prompt)

    # Remove rows like "Average" or blank rows.
    df = df[df["Prompt"].notna()]
    df = df[~df["Prompt"].str.lower().isin(["average", "averages", "avg", "nan"])]

    return df


def find_metric_column(df: pd.DataFrame, system: str, metric: str) -> str:
    system_key = system.lower().replace(" ", "")
    metric_key = metric.lower().replace(" ", "")

    candidates = []

    for col in df.columns:
        col_key = str(col).lower().replace(" ", "")

        if system_key in col_key and metric_key in col_key:
            candidates.append(col)

    if not candidates:
        raise ValueError(
            f"Could not find column for system={system!r}, metric={metric!r}.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return candidates[0]


def compute_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    metric_columns = {
        system: {
            metric: find_metric_column(df, system, metric)
            for metric in METRICS
        }
        for system in SYSTEMS
    }

    for category, prompts in PROMPT_CATEGORIES.items():
        normalized_prompts = [normalize_prompt(p) for p in prompts]
        category_df = df[df["Prompt"].isin(normalized_prompts)].copy()

        if category_df.empty:
            print(f"Warning: no rows found for category {category!r}")
            continue

        missing_prompts = [
            prompt
            for prompt in normalized_prompts
            if prompt not in set(category_df["Prompt"])
        ]

        if missing_prompts:
            print(f"Warning: missing prompts for category {category!r}:")
            for prompt in missing_prompts:
                print(f"  - {prompt}")

        for system in SYSTEMS:
            row = {
                "Category": category,
                "System": system,
            }

            for metric in METRICS:
                col = metric_columns[system][metric]
                value = pd.to_numeric(category_df[col], errors="coerce").mean()

                if metric == "Macro Usage":
                    value *= 100

                row[metric] = value

            rows.append(row)

    return pd.DataFrame(rows)


def format_number(value: float, bold: bool) -> str:
    if pd.isna(value):
        formatted = "--"
    else:
        formatted = f"{value:.2f}"

    if bold and formatted != "--":
        return r"\textbf{" + formatted + "}"

    return formatted


def make_latex_table(summary: pd.DataFrame) -> str:
    lines = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Summary of ablation experiments for each prompt category, averaged over all trials and environments.}"
    )
    lines.append(r"\label{tab:ablation_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Prompt Category & System & Success Rate & Program Length & Macro Usage (\%) & Runtime (s) \\"
    )
    lines.append(r"\midrule")

    categories = list(PROMPT_CATEGORIES.keys())

    for category_idx, category in enumerate(categories):
        category_df = summary[summary["Category"] == category].copy()

        if category_df.empty:
            continue

        best_values = {}

        for metric in METRICS:
            if metric in DO_NOT_BOLD:
                continue

            if metric == "Runtime":
                best_values[metric] = category_df[metric].min()
            else:
                best_values[metric] = category_df[metric].max()

        for row_idx, (_, row) in enumerate(category_df.iterrows()):
            category_cell = (
                rf"\multirow{{{len(category_df)}}}{{*}}{{{latex_escape(category)}}}"
                if row_idx == 0
                else ""
            )

            values = []

            for metric in METRICS:
                value = row[metric]

                should_bold = (
                    metric not in DO_NOT_BOLD
                    and pd.notna(value)
                    and np.isclose(value, best_values[metric])
                )

                values.append(format_number(value, should_bold))

            line = (
                f"{category_cell} & {latex_escape(row['System'])} & "
                f"{values[0]} & {values[1]} & {values[2]} & {values[3]} \\\\"
            )
            lines.append(line)

        if category_idx != len(categories) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    excel_path = find_excel_file()
    print(f"Reading {excel_path}")

    df = load_results_sheet(excel_path)
    summary = compute_category_summary(df)

    latex = make_latex_table(summary)

    OUTPUT_FILE.write_text(latex)

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
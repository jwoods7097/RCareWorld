from pathlib import Path
import re
import json
import html

import pandas as pd
import gradio as gr


RESULTS_DIR = Path(__file__).parent / "results"


def csv_timestamp(path: Path):
    """
    Extract timestamp from filenames like:
    user_20260622_111739.csv
    """
    match = re.search(r"(\d{8}_\d{6})", path.name)
    return match.group(1) if match else ""


def latest_two_csvs():
    csvs = sorted(
        RESULTS_DIR.glob("*.csv"),
        key=csv_timestamp
    )

    if len(csvs) < 2:
        raise FileNotFoundError("Need at least 2 CSV files in the results folder.")

    return csvs[-2], csvs[-1]


def extract_functions(code_value):
    """
    Extract unique `function` values from the JSONL in the `code` column.

    Handles:
    - one JSON object per line
    - a single JSON object
    - a JSON list
    - regex fallback
    """
    if pd.isna(code_value):
        return []

    text = str(code_value).strip()
    functions = []

    # Try JSONL first: one JSON object per line
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)

            if isinstance(obj, dict) and "function" in obj:
                functions.append(obj["function"])

            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and "function" in item:
                        functions.append(item["function"])

        except Exception:
            pass

    # Try the whole cell as JSON
    if not functions:
        try:
            obj = json.loads(text)

            if isinstance(obj, dict) and "function" in obj:
                functions.append(obj["function"])

            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and "function" in item:
                        functions.append(item["function"])

        except Exception:
            pass

    # Fallback for text containing: "function": "some_name"
    if not functions:
        functions = re.findall(r'"function"\s*:\s*"([^"]+)"', text)

    # Preserve order while making unique
    seen = set()
    unique = []

    for fn in functions:
        if fn not in seen:
            seen.add(fn)
            unique.append(fn)

    return unique


def summarize_csv(csv_path):
    df = pd.read_csv(csv_path)

    if "prompt" not in df.columns:
        raise ValueError(f"{csv_path.name} does not contain a 'prompt' column.")

    if "code" not in df.columns:
        raise ValueError(f"{csv_path.name} does not contain a 'code' column.")

    grouped = {}

    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        functions = extract_functions(row["code"])

        if prompt not in grouped:
            grouped[prompt] = []

        for fn in functions:
            if fn not in grouped[prompt]:
                grouped[prompt].append(fn)

    return grouped


def render_conversation(title, csv_path, grouped):
    parts = [
        f"""
        <div class="panel">
            <div class="panel-title">{html.escape(title)}</div>
            <div class="scroll-box">
        """
    ]

    for prompt, functions in grouped.items():
        safe_prompt = html.escape(prompt)

        if functions:
            function_html = "".join(
                f'<span class="chip">{html.escape(fn)}</span>'
                for fn in functions
            )
        else:
            function_html = '<span class="muted">No function values found</span>'

        parts.append(
            f"""
            <div class="turn">
                <div class="prompt-bubble">
                    <div class="label">Prompt</div>
                    {safe_prompt}
                </div>

                <div class="function-bubble">
                    <div class="label">Skills Used</div>
                    <div class="chips">{function_html}</div>
                </div>
            </div>
            """
        )

    parts.append("</div></div>")
    return "\n".join(parts)


def load_latest():
    try:
        csv_a, csv_b = latest_two_csvs()

        data_a = summarize_csv(csv_a)
        data_b = summarize_csv(csv_b)

        html_a = render_conversation("System A", csv_a, data_a)
        html_b = render_conversation("System B", csv_b, data_b)

        return html_a, html_b

    except Exception as e:
        error_html = f"""
        <div class="panel">
            <div class="panel-title">Error</div>
            <div class="scroll-box">
                <div class="prompt-bubble">{html.escape(str(e))}</div>
            </div>
        </div>
        """
        return error_html, error_html


css = """
.container {
    max-width: 1400px;
    margin: auto;
    color: black;
}

.panel {
    border: 1px solid #ddd;
    border-radius: 14px;
    padding: 14px;
    background: #fafafa;
    color: black;
}

.panel-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 4px;
    color: black;
}

.file-name {
    font-size: 13px;
    color: black;
    margin-bottom: 12px;
}

.scroll-box {
    height: 650px;
    overflow-y: auto;
    padding-right: 8px;
    color: black;
}

.turn {
    margin-bottom: 18px;
}

.function-bubble {
    background: white;
    border: 1px solid #e5e5e5;
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 15px;
    color: black;
}

.prompt-bubble {
    background: #f1f1f1;
    border-radius: 14px;
    padding: 12px 14px;
    margin-left: 22px;
    color: black;
}

.label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: black;
    margin-bottom: 6px;
    font-weight: 700;
}

.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.chip {
    display: inline-block;
    border: 1px solid #ccc;
    border-radius: 999px;
    padding: 4px 9px;
    background: white;
    font-family: monospace;
    font-size: 13px;
    color: black;
}

.muted {
    color: black;
    font-style: italic;
}
"""


with gr.Blocks(css=css) as demo:
    gr.Markdown("# Compare Systems")

    with gr.Row(elem_classes=["container"]):
        with gr.Column():
            system_a = gr.HTML()

        with gr.Column():
            system_b = gr.HTML()

    refresh = gr.Button("Reload")

    demo.load(load_latest, outputs=[system_a, system_b])
    refresh.click(load_latest, outputs=[system_a, system_b])


if __name__ == "__main__":
    demo.launch()
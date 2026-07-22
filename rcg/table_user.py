from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev
from datetime import time
from openpyxl import load_workbook

BASE_DIR = Path(__file__).resolve().parent
INPUT_XLSX = BASE_DIR / 'results' / 'User Study Results.xlsx'
OUTPUT_TEX = BASE_DIR / 'plots' / 'user_results.tex'

TASKS = [
    'Rearrange Objects',
    'Meal Preparation',
    'Feed Care Recipient',
]
SYSTEMS = [
    ('Baseline', [(2, 3, 4), (5, 6, 7), (8, 9, 10)]),
    ('SkillComposer', [(11, 12, 13), (14, 15, 16), (17, 18, 19)]),
]


def as_seconds(v) -> float:
    """Convert an Excel time cell or a numeric duration to seconds."""
    if isinstance(v, time):
        return v.hour * 3600 + v.minute * 60 + v.second
    if isinstance(v, (int, float)):
        return float(v)
    raise TypeError(f'Unsupported time value: {v!r}')


def fmt_h_mm(seconds: float) -> str:
    total = int(round(seconds))
    sign = '-' if total < 0 else ''
    total = abs(total)
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    return f'{sign}{hours}:{minutes:02d}'


def summarize(values, kind: str) -> tuple[str, float]:
    m = mean(values)
    s = stdev(values) if len(values) > 1 else 0.0
    if kind == 'time':
        text = f'{fmt_h_mm(m)} $\\pm$ {fmt_h_mm(s)}'
    elif kind == 'success':
        text = f'{m:.2f} $\\pm$ {s:.2f}'
    else:
        text = f'{m:.1f} $\\pm$ {s:.1f}'
    return text, m


def main() -> None:
    wb = load_workbook(INPUT_XLSX, data_only=True)
    ws = wb[wb.sheetnames[0]]

    # Keep only non-pilot participants.
    participant_rows = []
    for r in range(4, ws.max_row + 1):
        participant = ws.cell(r, 1).value
        reimbursed = ws.cell(r, 21).value
        if participant and reimbursed == 'Yes':
            participant_rows.append(r)

    metrics = ['time', 'success', 'prompts']
    summaries = {}
    means = {}

    for system_name, task_cols in SYSTEMS:
        summaries[system_name] = []
        means[system_name] = []
        for col_triplet in task_cols:
            task_summary = {}
            task_means = {}
            # Time
            time_vals = [as_seconds(ws.cell(r, col_triplet[0]).value) for r in participant_rows]
            text, m = summarize(time_vals, 'time')
            task_summary['time'] = text
            task_means['time'] = m
            # Success
            success_vals = [1.0 if ws.cell(r, col_triplet[1]).value == 'Yes' else 0.0 for r in participant_rows]
            text, m = summarize(success_vals, 'success')
            task_summary['success'] = text
            task_means['success'] = m
            # Prompts
            prompt_vals = [float(ws.cell(r, col_triplet[2]).value) for r in participant_rows]
            text, m = summarize(prompt_vals, 'prompts')
            task_summary['prompts'] = text
            task_means['prompts'] = m

            summaries[system_name].append(task_summary)
            means[system_name].append(task_means)

    # Determine best system per task and metric.
    best = {system: [dict() for _ in TASKS] for system, _ in SYSTEMS}
    for task_idx in range(len(TASKS)):
        # Higher is better for success; lower is better for time and prompts.
        for metric in metrics:
            a_name, b_name = SYSTEMS[0][0], SYSTEMS[1][0]
            a = means[a_name][task_idx][metric]
            b = means[b_name][task_idx][metric]
            if metric == 'success':
                if a >= b:
                    best[a_name][task_idx][metric] = True
                if b >= a:
                    best[b_name][task_idx][metric] = True
            else:
                if a <= b:
                    best[a_name][task_idx][metric] = True
                if b <= a:
                    best[b_name][task_idx][metric] = True

    lines = []
    lines.append('\\begin{table*}[t]')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append('\\caption{Summary of user study task metrics, averaged over all participants.}')
    lines.append('\\label{tab:user_results}')
    lines.append('\\begin{tabular}{llccc}')
    lines.append('\\toprule')
    lines.append('System & Task & Success Rate & Num. Prompts & Task Time (m:ss) \\\\')
    lines.append('\\midrule')

    for system_idx, (system_name, _) in enumerate(SYSTEMS):
        for task_idx, task_name in enumerate(TASKS):
            prefix = f'\\multirow{{3}}{{*}}{{{system_name}}}' if task_idx == 0 else ''
            cells = []
            for metric in metrics:
                text = summaries[system_name][task_idx][metric]
                if best[system_name][task_idx].get(metric):
                    text = f'\\textbf{{{text}}}'
                cells.append(text)
            row = f'{prefix} & {task_name} & {cells[1]} & {cells[2]} & {cells[0]} \\\\'
            lines.append(row)
        if system_idx == 0:
            lines.append('\\midrule')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table*}')
    lines.append('')

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {OUTPUT_TEX}')


if __name__ == '__main__':
    main()
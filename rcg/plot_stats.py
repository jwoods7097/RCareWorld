import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = "results/User Study (Responses).xlsx"
SHEET = "Numeric Responses"
OUTPUT_FILE = "plots/stats.pdf"

df = pd.read_excel(FILE, sheet_name=SHEET)

sus_questions = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system well-integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]

other_questions = [
    "Rate how effective the system was in understanding and executing the commands",
    "Rate how easy it was to predict the outcome of the system",
    "Rate how well the robot’s used skills aligned with the task",
    "Rate your overall satisfaction with the system",
]

def find_column(system_prefix, question_text):
    pattern = re.escape(question_text)
    matches = [c for c in df.columns if c.startswith(system_prefix) and re.search(pattern, c)]
    if not matches:
        raise ValueError(f"Could not find column for {system_prefix} / {question_text}")
    return matches[0]

def compute_sus_score(row, system_prefix):
    total = 0.0
    for i, question in enumerate(sus_questions, start=1):
        col = find_column(system_prefix, question)
        response = row[col]
        if pd.isna(response):
            return None

        response = float(response)

        if i in [1, 3, 5, 7, 9]:
            total += response - 1
        else:
            total += 7 - response

    return total * (100.0 / 60.0)

def mean_and_2sem(values):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan, np.nan, 0
    mean = vals.mean()
    if len(vals) == 1:
        return mean, np.nan, 1
    sem2 = 2 * vals.std(ddof=1) / np.sqrt(len(vals))
    return mean, sem2, len(vals)

def wrap_labels(labels, width=24):
    return ["\n".join(textwrap.wrap(label, width=width)) for label in labels]

# ----------------------------
# SUS scores per participant
# ----------------------------
baseline_sus = df.apply(lambda row: compute_sus_score(row, "Baseline"), axis=1)
skill_sus = df.apply(lambda row: compute_sus_score(row, "SkillComposer"), axis=1)

sus_pairs = pd.DataFrame({
    "Baseline": baseline_sus,
    "SkillComposer": skill_sus,
}).dropna()

sus_baseline_mean, sus_baseline_err, sus_n = mean_and_2sem(sus_pairs["Baseline"])
sus_skill_mean, sus_skill_err, _ = mean_and_2sem(sus_pairs["SkillComposer"])

# ----------------------------
# Other 4 questions
# ----------------------------
other_rows = []
for question in other_questions:
    bcol = find_column("Baseline", question)
    scol = find_column("SkillComposer", question)

    bmean, berr, n_b = mean_and_2sem(df[bcol])
    smean, serr, n_s = mean_and_2sem(df[scol])

    other_rows.append({
        "question": question,
        "baseline_mean": bmean,
        "baseline_err": berr,
        "skill_mean": smean,
        "skill_err": serr,
        "n_pairs": min(n_b, n_s),
    })

# ----------------------------
# Plot
# ----------------------------
fig, (ax_top, ax_bottom) = plt.subplots(
    2,
    1,
    figsize=(11, 8),
    gridspec_kw={"height_ratios": [1, 4], "hspace": 0.12}
)

# Global title + legend
fig.suptitle("Post-Questionnaire Ratings", y=0.98, fontsize=24)

# ----------------------------
# Top panel: SUS
# ----------------------------
row_pitch = 1.0
bar_h = 0.34
offset = 0.18

y_top = np.array([0.0])

ax_top.barh(
    y_top - offset,
    [sus_baseline_mean],
    height=bar_h,
    xerr=[sus_baseline_err],
    capsize=4,
    label="Baseline",
)

ax_top.barh(
    y_top + offset,
    [sus_skill_mean],
    height=bar_h,
    xerr=[sus_skill_err],
    capsize=4,
    label="SkillComposer",
)

ax_top.set_yticks([0])
ax_top.set_yticklabels(["SUS"], fontsize=16)
ax_top.set_ylim(0.5, -0.5)
ax_top.set_xlim(0, 100)
ax_top.set_xticks(range(0, 101, 10))
ax_top.margins(y=0)

# ----------------------------
# Bottom panel: other 4 questions
# ----------------------------
labels = ["Effectiveness", "Predictability", "Skill Alignment", "Satisfaction"]
wrapped_labels = labels #wrap_labels(labels, width=28)

baseline_means = [row["baseline_mean"] for row in other_rows]
baseline_errs = [row["baseline_err"] for row in other_rows]
skill_means = [row["skill_mean"] for row in other_rows]
skill_errs = [row["skill_err"] for row in other_rows]

y = np.arange(len(labels)) * row_pitch

ax_bottom.barh(
    y - offset,
    baseline_means,
    height=bar_h,
    xerr=baseline_errs,
    capsize=4,
    label="Baseline",
)

ax_bottom.barh(
    y + offset,
    skill_means,
    height=bar_h,
    xerr=skill_errs,
    capsize=4,
    label="SkillComposer",
)

ax_bottom.set_yticks(y)
ax_bottom.set_yticklabels(wrapped_labels, fontsize=16)
ax_bottom.set_ylim(y[-1] + 0.5, y[0] - 0.5)
ax_bottom.margins(y=0)

# Optional: make the x-scale comparable across panels
ax_bottom.set_xlim(1, 7)
ax_bottom.set_xticks(range(1, 8))

# One shared legend above both plots
handles, labels = ax_top.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.925),
    ncol=2,
    frameon=False,
    fontsize=12,
)

plt.subplots_adjust(left=0.2)
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()
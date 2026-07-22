import re
import pandas as pd
from scipy.stats import ttest_rel

FILE = "results/User Study (Responses).xlsx"
SHEET = "Numeric Responses"
OUTPUT_FILE = "results/stats_results.xlsx"

df = pd.read_excel(FILE, sheet_name=SHEET)

# ----------------------------
# Question definitions
# ----------------------------
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

all_questions = [("SUS", sus_questions)] + [(q, [q]) for q in other_questions]

# ----------------------------
# Helpers
# ----------------------------
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
            return None  # skip incomplete rows

        response = float(response)

        # SUS scoring
        if i in [1, 3, 5, 7, 9]:
            total += response - 1
        else:
            total += 7 - response

    return total * (100.0 / 60.0)

def summarize_paired(baseline_values, skill_values):
    paired = pd.DataFrame({
        "baseline": pd.to_numeric(baseline_values, errors="coerce"),
        "skillcomposer": pd.to_numeric(skill_values, errors="coerce"),
    }).dropna()

    n_pairs = len(paired)
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "baseline_mean": None,
            "skillcomposer_mean": None,
            "difference_mean": None,
            "difference_std": None,
            "t_statistic": None,
            "p_value": None,
        }

    diff = paired["skillcomposer"] - paired["baseline"]

    if n_pairs > 1:
        t_stat, p_value = ttest_rel(paired["baseline"], paired["skillcomposer"])

        diff_std = diff.std(ddof=1)
        cohen_d = diff.mean() / diff_std
    else:
        t_stat, p_value = None, None
        diff_std = None
        cohen_d = None

    return {
        "n_pairs": n_pairs,
        "baseline_mean": paired["baseline"].mean(),
        "baseline_std": paired["baseline"].std(ddof=1),
        "skillcomposer_mean": paired["skillcomposer"].mean(),
        "skillcomposer_std": paired["skillcomposer"].std(ddof=1),
        "difference_mean": diff.mean(),
        "difference_std": diff_std,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohen_d": cohen_d,
    }

# ----------------------------
# Build participant-level SUS scores
# ----------------------------
baseline_sus = df.apply(lambda row: compute_sus_score(row, "Baseline"), axis=1)
skill_sus = df.apply(lambda row: compute_sus_score(row, "SkillComposer"), axis=1)

results = []

# SUS row
sus_summary = summarize_paired(baseline_sus, skill_sus)
sus_summary["question"] = "SUS"
results.append(sus_summary)

# Other 4 questions
for question in other_questions:
    baseline_col = find_column("Baseline", question)
    skill_col = find_column("SkillComposer", question)

    summary = summarize_paired(df[baseline_col], df[skill_col])
    summary["question"] = question
    results.append(summary)

# ----------------------------
# Save results
# ----------------------------
results_df = pd.DataFrame(results, columns=[
    "question",
    "n_pairs",
    "baseline_mean",
    "baseline_std",
    "skillcomposer_mean",
    "skillcomposer_std",
    "difference_mean",
    "difference_std",
    "t_statistic",
    "p_value",
    "cohen_d",
])

results_df.to_excel(OUTPUT_FILE, index=False)

print(f"Saved results to {OUTPUT_FILE}")

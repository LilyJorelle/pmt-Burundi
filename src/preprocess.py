"""preprocess.py: apply the decisions of notebook 01 (docs/decision_log.md) to the raw data.

Reads the raw data through ingest.py, applies each cleaning rule in order,
and saves one clean table in data/processed/.
"""

import numpy as np
import pandas as pd

from ingest import MISSING_CODE, ROOT, load_codebook, load_data

# Input and output files. The processed table stays out of Git, like the raw data.
PREDICTORS_FILE = ROOT / "docs" / "final_predictors.txt"
OUTPUT_FILE = ROOT / "data" / "processed" / "clean.csv"

# Step 3: keep only households whose interview was filled (hh9 = 1).
VISIT_COLUMN, FILLED = "hh9", 1

# Step 4: poor = 1 when the item count (m11cons) is at or below the 40th percentile of the kept households.
TARGET_SOURCE, POOR_QUANTILE = "m11cons", 0.40

# Step 6, Part 4: after emptying the "Not known" codes, two codes hide real categories inside numbers.
SPECIAL_RECODES = {"ws3": {995: 0},   # 995 = water on the premises, so 0 minutes
                   "tn5": {95: 25}}   # 95 = more than 24 months, so 25

# Step 6, Part 10: a count skipped because the gate answer is No (2) is really 0.
GATE_ZERO_FILL = {"hc13": [f"hc14_{i}" for i in range(1, 9)],  # no livestock
                  "hc11": ["hc12"],                           # no cultivable land
                  "tn1": ["tn2", "tn3"]}                      # no mosquito nets
NO = 2

# The cluster number is kept next to the predictors for the grouped train/test split, but it is never a predictor.
CLUSTER_COLUMN = "hh1"



def keep_filled(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: keep only households whose interview was filled (hh9 = 1)."""
    # interview_status is a review workflow, not completion, so the visit result hh9 decides who stays.
    kept = df[df[VISIT_COLUMN] == FILLED].copy()

    # Notebook 01 showed that the target is complete for every filled household; stop loudly if that ever changes.
    assert kept[TARGET_SOURCE].notna().all(), f"{TARGET_SOURCE} has missing values among filled interviews"
    return kept



def build_label(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Step 4: add the label poor = 1 when m11cons is at or below the 40th percentile of the kept households."""
    # The cut-off comes from the kept households only, so it is computed after keep_filled().
    cutoff = df[TARGET_SOURCE].quantile(POOR_QUANTILE)

    # Work on a copy and store the label as a whole number (1 = poor, 0 = not poor).
    out = df.copy()
    out["poor"] = (out[TARGET_SOURCE] <= cutoff).astype(int)
    return out, cutoff


# Step 6, Part 2: the 18 variables that mix real counts or quantities with special codes.
COUNTS_WITH_CODES = (["hc9_7", "hc9_9", "hc9_13", "hc9_14", "hc9_16", "hc9_23", "hc9_24", "hc12"]
                     + [f"hc14_{i}" for i in range(1, 9)] + ["ws3", "tn5"])

# Step 6, Part 7: the ceiling rule applies to the housing, water and net variables.
CEILING_PATTERN = r"^(hc|ws|tn)\d"


def clean_special_codes(df: pd.DataFrame, value_labels: dict) -> tuple[pd.DataFrame, int]:
    """Step 6, Parts 3 to 7: empty the "Not known" codes and impossible values, then recode two hidden categories."""
    out = df.copy()
    before = int(out.isna().sum().sum())

    # Highest labelled code of a variable, ignoring the -999999999 legend (None if it has no value labels).
    def ceiling(col):
        real = [k for k in value_labels.get(col, {}) if k > MISSING_CODE]
        return max(real) if real else None

    # Rule 1: in the 18 mixed variables the highest code means "Not known" (98 or 998), so it becomes empty.
    # The "97 and above" code stays, because it is a real, capped count.
    for col in COUNTS_WITH_CODES:
        out.loc[out[col] >= ceiling(col), col] = np.nan

    # Rule 2: any value above the highest labelled code is a data-entry error (for example 30,000,000 cars).
    numeric = out.select_dtypes("number").columns
    for col in numeric[numeric.str.match(CEILING_PATTERN)]:
        if ceiling(col) is not None:
            out.loc[out[col] > ceiling(col), col] = np.nan

    # Rule 3: two real categories hide inside numeric columns.
    for col, codes in SPECIAL_RECODES.items():
        out[col] = out[col].replace(codes)

    # Return the number of cells that became empty, to compare with notebook 01.
    return out, int(out.isna().sum().sum()) - before



def fill_skipped_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6, Part 10: a count skipped because the gate answer is No (2) is really 0."""
    out = df.copy()
    for gate, columns in GATE_ZERO_FILL.items():
        # Households that answered No to the gate question never saw the follow-up questions.
        skipped = out[gate] == NO
        for col in columns:
            # Fill only the empty cells of those households; a "Not known" answer stays empty.
            out.loc[skipped & out[col].isna(), col] = 0
    return out



def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 7: keep the cluster number, the 111 predictors and the label; everything else is excluded."""
    # The predictor list written by notebook 01, Step 9, Part 1 (one variable name per line).
    predictors = PREDICTORS_FILE.read_text(encoding="utf-8").splitlines()

    # Safety checks: every predictor exists, and neither the target source nor the cluster can be a predictor.
    missing = [c for c in predictors if c not in df.columns]
    assert not missing, f"Predictors missing from the data: {missing}"
    assert TARGET_SOURCE not in predictors, f"{TARGET_SOURCE} builds the label, it cannot be a predictor"
    assert CLUSTER_COLUMN not in predictors, "The cluster number is only used to group the split"

    # Cluster first and label last, so the table reads: grouping column, predictors, answer.
    return df[[CLUSTER_COLUMN] + predictors + ["poor"]]



def run() -> pd.DataFrame:
    """Run the whole pipeline: raw data in, clean table out."""
    # Load the codebook (only the value labels are needed here) and the raw data.
    _, value_labels = load_codebook()
    df = load_data()

    # Apply the decisions of notebook 01 in the same order as its steps.
    df = keep_filled(df)                              # Step 3
    df, cutoff = build_label(df)                      # Step 4
    df, emptied = clean_special_codes(df, value_labels)  # Step 6, Parts 3 to 7
    df = fill_skipped_counts(df)                      # Step 6, Part 10
    clean = select_columns(df)                        # Step 7

    # Save the clean table (create data/processed if it does not exist yet).
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT_FILE, index=False)

    # Summary only: counts and percentages, no household rows are printed.
    print(f"Households kept: {len(clean)}")
    print(f"Cut-off on {TARGET_SOURCE}: {cutoff:g} | poor: {int(clean['poor'].sum())} ({clean['poor'].mean():.1%})")
    print(f"Cells emptied by the rules: {emptied}")
    print(f"Table: {clean.shape[0]} rows x {clean.shape[1]} columns (predictors + cluster + label)")
    print(f"Empty cells among predictors and label: {clean.drop(columns=CLUSTER_COLUMN).isna().mean().mean():.1%}")
    print(f"Saved: {OUTPUT_FILE}")
    return clean


if __name__ == "__main__":
    # Runs only when the file is executed directly: python src/preprocess.py
    run()
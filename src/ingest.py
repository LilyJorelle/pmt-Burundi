"""ingest.py: read the raw household data and the translated codebook.

Functions:
    load_data()      returns the raw table with all 221 columns
    load_codebook()  returns the labels and value_labels dictionaries (notebook 01, Step 2)
"""

from pathlib import Path

import pandas as pd

# The project root is the folder above src/, so the script works from any location.
ROOT = Path(__file__).resolve().parents[1]

# File locations in one place: if a file is renamed, only this block changes.
RAW_DIR = ROOT / "data" / "raw"
DATA_FILE = RAW_DIR / "hh.csv"
CODEBOOK_FILE = RAW_DIR / "household codebook file - Translation.xlsx"

# Legend value the codebook uses for "missing" (it does not appear in the data itself).
MISSING_CODE = -999999999



# The translation gave hc16, hc17 and hc18 the same English question; the French originals ask about
# information, communication and transport (found while reading the top predictors of notebook 02).
STEM_FIXES = {"hc16_": "What means of information does the household own",
              "hc17_": "What means of communication does the household own",
              "hc18_": "What means of transport does the household own"}


def load_data() -> pd.DataFrame:
    """Read the raw household data exactly as stored: 221 columns, numeric codes, nothing changed."""
    # Fail early with a clear message if the file is not in data/raw (the data is restricted).
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Place the data file here first: {DATA_FILE}")

    # Same call as notebook 01 (Step 1, Part 1), so the table is identical.
    return pd.read_csv(DATA_FILE)  





# Names the codebook got wrong (found in notebook 01, Step 2, Part 3): the entry at position 35 lost its name.
NAME_FIXES = {"1": "hh17b", "1.0": "hh17b"}


def load_codebook() -> tuple[dict, dict]:
    """Return (labels, value_labels) built from the translated codebook.

    labels:       variable code -> English name, e.g. "hh6" -> "Residence Area"
    value_labels: variable code -> {value code: meaning}, e.g. "hh6" -> {1.0: "Rural", 2.0: "Urban"}
    """
    # Fail early with a clear message if the codebook is not in data/raw.
    if not CODEBOOK_FILE.exists():
        raise FileNotFoundError(f"Place the codebook file here first: {CODEBOOK_FILE}")

    # Read the sheet without a header row: row 0 is a title and row 1 holds the column headers.
    cb = pd.read_excel(CODEBOOK_FILE, sheet_name="Sheet1", header=None)
    cb.columns = ["variable", "position", "label", "translation", "extra"]
    cb = cb.iloc[2:].reset_index(drop=True)

    # Section 1: rows with a numeric position are variables; the first row without one starts the next section.
    cb["pos_num"] = pd.to_numeric(cb["position"], errors="coerce")
    first_break = cb["pos_num"].isna().idxmax()
    var_block = cb.iloc[:first_break].copy()
    var_block["variable"] = var_block["variable"].astype(str).replace(NAME_FIXES)

    # Build labels, using the English translation and falling back to the original label if it is empty.
    labels = dict(zip(var_block["variable"], var_block["translation"].fillna(var_block["label"])))

    # Section 2: the value labels start two rows after the "Variable Values" title (one row is a header).
    start = cb.index[cb["variable"] == "Variable Values"][0]
    vb = cb.iloc[start + 2:].copy()

    # The variable name appears only on the first row of each variable, so fill it downwards.
    vb["variable"] = vb["variable"].ffill().astype(str).replace(NAME_FIXES)
    vb = vb.dropna(subset=["position"])

    # Codes such as "-999999999,000a" become numbers: decimal comma to point, trailing letter removed.
    vb["code"] = pd.to_numeric(vb["position"].astype(str).str.replace(",", ".").str.rstrip("a"), errors="coerce")
    vb["text"] = vb["translation"].fillna(vb["label"])

    # Build value_labels: variable -> {code: meaning}.
    value_labels = {v: dict(zip(g["code"], g["text"])) for v, g in vb.groupby("variable")}

    # Replace the wrong question text for hc16, hc17 and hc18, and keep the answer option after the colon.
    for var in labels:
        for prefix, stem in STEM_FIXES.items():
            text = str(labels[var])
            if var.startswith(prefix) and ": " in text:
                labels[var] = f"{stem}: {text.split(': ', 1)[1]}"

    return labels, value_labels



if __name__ == "__main__":
    # Runs only when the file is executed directly (python src/ingest.py), not when other scripts import it.
    df = load_data()
    labels, value_labels = load_codebook()

    # Summary only: no household rows are printed.
    print(f"Data: {df.shape[0]} households, {df.shape[1]} columns")
    print(f"Codebook: {len(labels)} variable labels, {len(value_labels)} variables with value labels")

    # Every column in the data should have an English label.
    print("Columns without a label:", [c for c in df.columns if c not in labels])























































































































































































































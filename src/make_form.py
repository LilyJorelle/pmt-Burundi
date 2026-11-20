"""make_form.py: build api/form_fields.json, the list of questions the demo form shows.

Reads the chosen predictors (docs/short_predictors.csv), their wording and answer options
(docs/data_dictionary.csv) and their valid ranges (models/model_info.json). No household data is used.
"""

import json

import pandas as pd

from ingest import ROOT

SHORT_FILE = ROOT / "docs" / "short_predictors.csv"
DICT_FILE = ROOT / "docs" / "data_dictionary.csv"
INFO_FILE = ROOT / "models" / "model_info.json"
OUTPUT_FILE = ROOT / "api" / "form_fields.json"

# The order of the sections on the form: basics first, then the home, then goods and so on.
TOPIC_ORDER = ["Location", "Household head and size", "Dwelling and energy", "Water and sanitation",
               "Household goods", "Livestock, land and farm work", "Communication, information, transport",
               "Mosquito nets"]

# Clearer wording for the four "none of these" tick items (1 = ticked, 0 = not ticked).
LABEL_FIXES = {"hc16_1": "The household owns NO means of information",
               "hc17_1": "The household owns NO means of communication",
               "hc18_1": "The household owns NO means of transport",
               "hc18_6": "The household owns another means of transport (other)"}

# Short hints shown under some number boxes.
HINTS = {"hhage": "Age in years", "hh12": "Women aged 15 to 49 in the household",
         "ws9": "Number of households using this toilet", "tn5": "In months; enter 25 if more than 24 months",
         "hc12": "Ares of agricultural land (0 if none)"}


def parse_options(text):
    """'1 = Yes; 2 = No' becomes [{'value': 1, 'label': 'Yes'}, {'value': 2, 'label': 'No'}]."""
    options = []
    for part in str(text).split(";"):
        code, _, label = part.partition("=")
        try:
            number = float(code.strip())
        except ValueError:
            continue
        options.append({"value": int(number) if number.is_integer() else number, "label": label.strip()})
    return options


def field_kind(text, low, high):
    """yesno = a 1/0 tick, number = a count or measure, choice = a coded question with answer options."""
    if pd.isna(text):
        return "yesno" if (low, high) == (0, 1) else "number"
    if "and above" in str(text) or "More than" in str(text):
        return "number"          # counts such as '97 = 97 and above' are numbers, not choices
    return "choice"


def run():
    chosen = pd.read_csv(SHORT_FILE)
    dictionary = pd.read_csv(DICT_FILE).set_index("code")
    ranges = json.loads(INFO_FILE.read_text())["ranges"]

    fields = []
    for _, row in chosen.iterrows():
        code = row["code"]
        low, high = ranges[code]
        text = dictionary.loc[code, "values"]
        kind = field_kind(text, low, high)

        options = []
        if kind == "yesno":
            options = [{"value": 1, "label": "Yes"}, {"value": 0, "label": "No"}]
        elif kind == "choice":
            # Offer only the options the API will accept (inside the range seen in the clean data).
            options = [o for o in parse_options(text) if low <= o["value"] <= high]

        fields.append({"code": code, "topic": row["topic"], "kind": kind, "min": low, "max": high,
                       "label": LABEL_FIXES.get(code, str(dictionary.loc[code, "english_name"])),
                       "hint": HINTS.get(code, ""), "options": options})

    # Sections in the chosen order; inside a section, the strongest question first (the CSV order).
    fields.sort(key=lambda f: TOPIC_ORDER.index(f["topic"]))
    OUTPUT_FILE.write_text(json.dumps(fields, indent=2, ensure_ascii=False), encoding="utf-8")

    for f in fields:
        detail = f"{len(f['options'])} options" if f["options"] else f"range {f['min']:g} to {f['max']:g}"
        print(f"{f['code']:<9} {f['kind']:<7} {detail}")
    print(f"\nSaved {OUTPUT_FILE.name} with {len(fields)} fields")


if __name__ == "__main__":
    run()
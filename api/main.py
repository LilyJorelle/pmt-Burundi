"""main.py: serve the trained XGBoost model through a FastAPI web service.

Run from the project root:   python -m uvicorn api.main:app
Interactive page (the demo): http://127.0.0.1:8000/docs
"""

import json
from pathlib import Path
import xgboost as xgb

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse

# The project root is the folder above api/, so the service finds the model from any location.
ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "xgb_model.joblib"
INFO_FILE = ROOT / "models" / "model_info.json"

# Load the model and its info once, when the service starts (not at every request).
model = joblib.load(MODEL_FILE)
info = json.loads(INFO_FILE.read_text(encoding="utf-8"))
PREDICTORS = info["predictors"]  # the 111 predictors, in the order the model expects
RANGES = info["ranges"]  # smallest and largest valid value of each predictor

app = FastAPI(
    title="pmt-Burundi: household poverty screening",
    description="Estimates from observable household characteristics whether a household is in the poorest 40%. "
                "A screening aid, not a final decision.",
)


@app.get("/health")
def health():
    """Check that the service is running and show which model it serves."""
    return {"status": "ok", "technique": info["technique"], "predictors": len(PREDICTORS),
            "test_metrics": info["test_metrics"]}


THRESHOLD = 0.5  # same rule as notebook 02: poor if the predicted risk is 0.5 or more


class Household(BaseModel):
    """The answers of one household, as variable code -> value. Questions not answered are simply left out."""
    answers: dict[str, float | None] = Field(
        description="Variable code -> answer (codes and meanings: docs/data_dictionary.csv). Anything left out is unknown.")


# Four typical profiles for the demo, built from the test data (the usual answers of the households the model is surest about). No real household is used.
EXAMPLES = {
    "rural_poor": {"summary": "Rural household, poor profile", "value": {"answers": {'hc17_1': 1, 'hc9_9': 0, 'hc13': 2, 'hc6': 5, 'helevel': 1, 'hh6': 1, 'ws7': 23, 'tn6': 2, 'hc9_14': 0, 'hc16_1': 1, 'hc14_8': 0, 'hc6a': 5, 'prov5': 3, 'ws5': 2, 'tn4': 1, 'hhage': 52, 'hc9_24': 0, 'hc18_1': 1, 'hh18': 3, 'hc7': 1, 'hh7': 2, 'ws9': 0, 'tn5': 8, 'hh12': 1, 'hc9_15': 0, 'hc18_6': 0, 'hc1c': 2, 'ws7a': 2, 'hc12': 2, 'hhmstatus': 1}}},   # model risk 83%
    "urban_poor": {"summary": "Urban household, poor profile", "value": {"answers": {'hc17_1': 0, 'hc9_9': 0, 'hc13': 2, 'hc6': 4, 'helevel': 2, 'hh6': 2, 'ws7': 22, 'tn6': 2, 'hc9_14': 0, 'hc16_1': 1, 'hc14_8': 0, 'hc6a': 1, 'prov5': 2, 'ws5': 2, 'tn4': 1, 'hhage': 43, 'hc9_24': 0, 'hc18_1': 1, 'hh18': 3, 'hc7': 1, 'hh7': 5, 'ws9': 1, 'tn5': 5, 'hh12': 1, 'hc9_15': 0, 'hc18_6': 0, 'hc1c': 3, 'ws7a': 3, 'hc12': 0, 'hhmstatus': 1}}},   # model risk 30%
    "rural_better_off": {"summary": "Rural household, better-off profile", "value": {"answers": {'hc17_1': 0, 'hc9_9': 2, 'hc13': 1, 'hc6': 5, 'helevel': 2, 'hh6': 1, 'ws7': 22, 'tn6': 2, 'hc9_14': 1, 'hc16_1': 0, 'hc14_8': 0, 'hc6a': 7, 'prov5': 5, 'ws5': 2, 'tn4': 1, 'hhage': 42, 'hc9_24': 1, 'hc18_1': 1, 'hh18': 3, 'hc7': 1, 'hh7': 3, 'ws9': 0, 'tn5': 6, 'hh12': 1, 'hc9_15': 0, 'hc18_6': 0, 'hc1c': 2, 'ws7a': 2, 'hc12': 49, 'hhmstatus': 1}}},   # model risk 10%
    "urban_better_off": {"summary": "Urban household, better-off profile", "value": {"answers": {'hc17_1': 0, 'hc9_9': 3, 'hc13': 2, 'hc6': 4, 'helevel': 3, 'hh6': 2, 'ws7': 12, 'tn6': 2, 'hc9_14': 1, 'hc16_1': 0, 'hc14_8': 0, 'hc6a': 1, 'prov5': 2, 'ws5': 2, 'tn4': 1, 'hhage': 41, 'hc9_24': 0, 'hc18_1': 1, 'hh18': 3, 'hc7': 1, 'hh7': 3, 'ws9': 0, 'tn5': 6, 'hh12': 1, 'hc9_15': 2, 'hc18_6': 0, 'hc1c': 3, 'ws7a': 1, 'hc12': 0, 'hhmstatus': 1}}},   # model risk 1%
}

@app.post("/predict")
def predict(household: Household = Body(openapi_examples=EXAMPLES)):
    """Estimate whether a household is in the poorest 40% from the answers given."""
    answers = household.answers

    # Refuse variable codes the model does not know.
    unknown = [k for k in answers if k not in PREDICTORS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown variable codes: {unknown}")

    # Refuse values outside the range seen in the clean training data (typing errors, impossible values).
    problems = [f"{k} = {v:g} is outside the valid range ({RANGES[k][0]:g} to {RANGES[k][1]:g})"
                for k, v in answers.items() if v is not None and not RANGES[k][0] <= v <= RANGES[k][1]]
    if problems:
        raise HTTPException(status_code=422, detail=problems)

    # One row with all 111 predictors in the training order; unanswered ones stay empty, which XGBoost handles.
    row = pd.DataFrame([{c: answers.get(c) for c in PREDICTORS}], columns=PREDICTORS).astype(float)
    risk = float(model.predict_proba(row)[0, 1])

    given = sum(v is not None for v in answers.values())
    return {"label": "poor" if risk >= THRESHOLD else "not poor", "risk": round(risk, 3),
            "answers_given": given, "answers_missing": len(PREDICTORS) - given,
            "note": "Screening aid, not a final decision. The fewer answers given, the less reliable the estimate.",
            "why": explain(row, answers)}
FORM_FILE = Path(__file__).parent / "form_fields.json"
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def form_page():
    """The simple demo page (a form), served at the root address."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/form-data", include_in_schema=False)
def form_data():
    """The questions for the form (made by src/make_form.py) and the two synthetic example households."""
    fields = json.loads(FORM_FILE.read_text(encoding="utf-8"))
    examples = {k: {"label": v["summary"], "answers": v["value"]["answers"]} for k, v in EXAMPLES.items()}
    return {"fields": fields, "examples": examples}



FIELDS = {f["code"]: f for f in json.loads((Path(__file__).parent / "form_fields.json").read_text(encoding="utf-8"))}


def answer_text(code, value):
    """The answer in words: the option label for coded questions, the number for counts."""
    for option in FIELDS[code]["options"]:
        if option["value"] == value:
            return option["label"]
    return f"{value:g}"


def explain(row, answers, top=3):
    """Why this verdict: how strongly each GIVEN answer pushed the score towards poor or towards not poor."""
    # pred_contribs gives one number per predictor (TreeSHAP); the last number is the baseline, so we drop it.
    contributions = model.get_booster().predict(xgb.DMatrix(row), pred_contribs=True)[0][:-1]
    given = [(c, float(v)) for c, v in zip(PREDICTORS, contributions) if answers.get(c) is not None]

    def describe(item):
        code, push = item
        return {"question": FIELDS[code]["label"], "answer": answer_text(code, answers[code]), "push": round(push, 2)}

    towards_poor = sorted([i for i in given if i[1] > 0], key=lambda i: -i[1])[:top]
    towards_not_poor = sorted([i for i in given if i[1] < 0], key=lambda i: i[1])[:top]
    return {"towards_poor": [describe(i) for i in towards_poor],
            "towards_not_poor": [describe(i) for i in towards_not_poor]}
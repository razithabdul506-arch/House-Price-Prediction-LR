"""
Vercel serverless function: POST /api/predict

Loads the pre-trained Linear Regression model (see train_model.py) and
returns a price prediction for the house features sent in the request body.

Expected JSON body:
{
  "area": 7420,
  "bedrooms": 4,
  "bathrooms": 2,
  "stories": 3,
  "parking": 2,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "no",
  "hotwaterheating": "no",
  "airconditioning": "yes",
  "prefarea": "yes",
  "furnishingstatus": "furnished"   # "furnished" | "semi-furnished" | "unfurnished"
}
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.json")

model = joblib.load(MODEL_PATH)
with open(COLUMNS_PATH) as f:
    FEATURE_COLUMNS = json.load(f)

app = Flask(__name__)

NUMERIC_FIELDS = ["area", "bedrooms", "bathrooms", "stories", "parking"]
YES_NO_FIELDS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]
REQUIRED_FIELDS = NUMERIC_FIELDS + YES_NO_FIELDS + ["furnishingstatus"]


def build_feature_row(payload):
    """Turn raw form input into the one-hot-encoded row the model expects."""
    row = {col: 0 for col in FEATURE_COLUMNS}

    for field in NUMERIC_FIELDS:
        row[field] = float(payload[field])

    for field in YES_NO_FIELDS:
        col_name = f"{field}_yes"
        if col_name in row:
            row[col_name] = 1 if str(payload[field]).lower() == "yes" else 0

    furnishing = str(payload["furnishingstatus"]).lower()
    semi_col = "furnishingstatus_semi-furnished"
    unfurnished_col = "furnishingstatus_unfurnished"
    if furnishing == "semi-furnished" and semi_col in row:
        row[semi_col] = 1
    elif furnishing == "unfurnished" and unfurnished_col in row:
        row[unfurnished_col] = 1
    # "furnished" is the dropped baseline category -> both stay 0

    return pd.DataFrame([[row[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


@app.route("/")
def index():
    """Serve the form for local dev, so the page and the API share an origin.

    On Vercel this route is never reached - index.html is served statically
    from the project root and only /api/predict is routed to this function.
    """
    if not os.path.exists(os.path.join(PROJECT_ROOT, "index.html")):
        return jsonify({"error": "index.html not found next to api/"}), 404
    return send_from_directory(PROJECT_ROOT, "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        X = build_feature_row(payload)
        prediction = model.predict(X)[0]
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    return jsonify({"predicted_price": round(float(prediction), 2)})


# Local dev entrypoint: `python api/predict.py` serves the form AND the API
# together on http://127.0.0.1:5000 - open that URL, not the index.html file.
if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)

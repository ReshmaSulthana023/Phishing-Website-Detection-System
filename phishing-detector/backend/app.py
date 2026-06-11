"""
app.py
Flask REST API for the AI-Powered Phishing Website Detection System.

Endpoints:
    POST /api/check          -> analyze a URL and return prediction
    GET  /api/history         -> list previously checked URLs (with search/filter)
    DELETE /api/history/<id>  -> delete a single history entry
    DELETE /api/history       -> clear all history
    GET  /api/dashboard        -> aggregate statistics for the dashboard
"""

import json
import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, request, jsonify, g, send_file
from flask_cors import CORS

from feature_extractor import (
    extract_features,
    features_to_vector,
    generate_reasons,
    FEATURE_NAMES,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ---------------------------------------------------------------------------
# Load ML model + metadata
# ---------------------------------------------------------------------------

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH) as f:
    MODEL_METADATA = json.load(f)

MODEL_ACCURACY = MODEL_METADATA.get("accuracy", 0.0)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            safety_score INTEGER NOT NULL,
            reasons TEXT NOT NULL,
            features TEXT NOT NULL,
            checked_at TEXT NOT NULL
        )
    """)
    db.commit()
    db.close()


# ---------------------------------------------------------------------------
# Helper: risk level + safety score
# ---------------------------------------------------------------------------

def compute_risk_level(prediction: int, phishing_prob: float) -> str:
    if prediction == 1:
        if phishing_prob >= 0.80:
            return "High"
        elif phishing_prob >= 0.55:
            return "Medium"
        else:
            return "Low"
    else:
        # Legitimate prediction -- still flag borderline confidence
        if phishing_prob >= 0.35:
            return "Medium"
        return "Low"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_file(os.path.join(FRONTEND_DIR, "index.html"))


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_accuracy": MODEL_ACCURACY})


@app.route("/api/check", methods=["POST"])
def check_url():
    data = request.get_json(silent=True) or {}
    raw_url = (data.get("url") or "").strip()

    if not raw_url:
        return jsonify({"error": "Please provide a URL to check."}), 400

    try:
        features = extract_features(raw_url)
    except Exception as exc:
        return jsonify({"error": f"Could not parse URL: {exc}"}), 400

    vector = pd.DataFrame([features_to_vector(features)], columns=FEATURE_NAMES)
    proba = model.predict_proba(vector)[0]  # [prob_legit, prob_phishing]
    prediction = int(model.predict(vector)[0])

    phishing_prob = float(proba[1])
    legit_prob = float(proba[0])

    confidence = phishing_prob if prediction == 1 else legit_prob
    confidence_pct = round(confidence * 100, 2)

    safety_score = round((1 - phishing_prob) * 100)
    risk_level = compute_risk_level(prediction, phishing_prob)
    reasons = generate_reasons(raw_url, features, prediction)

    label = "Phishing Website" if prediction == 1 else "Safe Website"

    result = {
        "url": raw_url,
        "prediction": label,
        "is_phishing": bool(prediction),
        "confidence": confidence_pct,
        "safety_score": safety_score,
        "risk_level": risk_level,
        "reasons": reasons,
        "features": features,
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }

    # Persist to history
    db = get_db()
    db.execute(
        """INSERT INTO checks
           (url, prediction, confidence, risk_level, safety_score, reasons, features, checked_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            raw_url,
            label,
            confidence_pct,
            risk_level,
            safety_score,
            json.dumps(reasons),
            json.dumps(features),
            result["checked_at"],
        ),
    )
    db.commit()
    result["id"] = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def get_history():
    search = request.args.get("search", "").strip().lower()
    risk_filter = request.args.get("risk", "").strip()  # Low / Medium / High
    pred_filter = request.args.get("prediction", "").strip()  # Safe / Phishing
    limit = request.args.get("limit", default=100, type=int)

    db = get_db()
    query = "SELECT * FROM checks WHERE 1=1"
    params = []

    if search:
        query += " AND LOWER(url) LIKE ?"
        params.append(f"%{search}%")

    if risk_filter:
        query += " AND risk_level = ?"
        params.append(risk_filter)

    if pred_filter:
        if pred_filter.lower() == "phishing":
            query += " AND prediction = 'Phishing Website'"
        elif pred_filter.lower() == "safe":
            query += " AND prediction = 'Safe Website'"

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = db.execute(query, params).fetchall()

    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "url": row["url"],
            "prediction": row["prediction"],
            "confidence": row["confidence"],
            "risk_level": row["risk_level"],
            "safety_score": row["safety_score"],
            "reasons": json.loads(row["reasons"]),
            "features": json.loads(row["features"]),
            "checked_at": row["checked_at"],
        })

    return jsonify({"history": history, "count": len(history)})


@app.route("/api/history/<int:check_id>", methods=["DELETE"])
def delete_history_item(check_id):
    db = get_db()
    db.execute("DELETE FROM checks WHERE id = ?", (check_id,))
    db.commit()
    return jsonify({"deleted": check_id})


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    db = get_db()
    db.execute("DELETE FROM checks")
    db.commit()
    return jsonify({"cleared": True})


@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    db = get_db()

    total = db.execute("SELECT COUNT(*) AS c FROM checks").fetchone()["c"]
    safe = db.execute(
        "SELECT COUNT(*) AS c FROM checks WHERE prediction = 'Safe Website'"
    ).fetchone()["c"]
    phishing = db.execute(
        "SELECT COUNT(*) AS c FROM checks WHERE prediction = 'Phishing Website'"
    ).fetchone()["c"]

    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    for row in db.execute(
        "SELECT risk_level, COUNT(*) AS c FROM checks GROUP BY risk_level"
    ).fetchall():
        risk_counts[row["risk_level"]] = row["c"]

    recent_rows = db.execute(
        "SELECT * FROM checks ORDER BY id DESC LIMIT 10"
    ).fetchall()
    recent = []
    for row in recent_rows:
        recent.append({
            "id": row["id"],
            "url": row["url"],
            "prediction": row["prediction"],
            "confidence": row["confidence"],
            "risk_level": row["risk_level"],
            "safety_score": row["safety_score"],
            "checked_at": row["checked_at"],
        })

    # Trend: counts grouped by date
    trend_rows = db.execute(
        """SELECT substr(checked_at, 1, 10) AS day,
                  SUM(CASE WHEN prediction = 'Phishing Website' THEN 1 ELSE 0 END) AS phishing,
                  SUM(CASE WHEN prediction = 'Safe Website' THEN 1 ELSE 0 END) AS safe
           FROM checks
           GROUP BY day
           ORDER BY day ASC
           LIMIT 30"""
    ).fetchall()
    trend = [
        {"date": row["day"], "phishing": row["phishing"], "safe": row["safe"]}
        for row in trend_rows
    ]

    return jsonify({
        "total_checked": total,
        "safe_count": safe,
        "phishing_count": phishing,
        "risk_distribution": risk_counts,
        "model_accuracy": round(MODEL_ACCURACY * 100, 2),
        "recent_checks": recent,
        "trend": trend,
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify(MODEL_METADATA)


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)

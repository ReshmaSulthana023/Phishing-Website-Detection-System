# 🛡️ ShieldScan — AI-Powered Phishing Website Detection System

A full-stack web application that uses a **Random Forest** machine learning
model to classify URLs as **Safe** or **Phishing**, with confidence scores,
a 0–100 safety score, and human-readable explanations of risk factors.

---

## 📁 Project Structure

```
phishing-detector/
├── backend/
│   ├── app.py                 # Flask REST API
│   ├── feature_extractor.py   # URL feature extraction + explanation logic
│   ├── train_model.py         # Dataset generation + model training script
│   ├── requirements.txt
│   ├── phishing_model.pkl     # Trained Random Forest model (generated)
│   ├── model_metadata.json    # Model accuracy + feature list (generated)
│   ├── dataset.csv            # Training dataset (generated)
│   └── database.db            # SQLite database (auto-created on first run)
└── frontend/
    └── index.html             # React + Tailwind single-page app (CDN-based)
```

---

## ⚙️ Setup Instructions

### 1. Backend (Flask + ML)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Train the model** (generates `dataset.csv`, `phishing_model.pkl`, `model_metadata.json`):

```bash
python3 train_model.py
```

**Run the API server**:

```bash
python3 app.py
```

The API will be available at `http://localhost:5000`. The SQLite database
(`database.db`) is created automatically on first run.

### 2. Frontend (React + Tailwind)

The frontend is a single self-contained HTML file (`frontend/index.html`)
that loads React, Tailwind CSS, Chart.js, and jsPDF from CDNs — no build
step required.

Simply open `frontend/index.html` in a browser, or serve it:

```bash
cd frontend
python3 -m http.server 8080
```

Then visit `http://localhost:8080`. Make sure the backend is running on
port 5000 (the frontend calls `http://localhost:5000/api`).

---

## 🧠 Machine Learning Pipeline

### Feature Extraction (`feature_extractor.py`)

For every URL, 13 features are extracted:

| Feature | Description |
|---|---|
| `url_length` | Total length of the URL |
| `domain_length` | Length of the domain/host |
| `num_dots` | Number of `.` characters |
| `num_hyphens` | Number of `-` characters |
| `num_subdomains` | Number of subdomain levels |
| `has_at_symbol` | Whether `@` appears in the URL |
| `has_https` | Whether the scheme is HTTPS |
| `has_ip` | Whether the host is a raw IP address |
| `num_digits` | Count of digit characters |
| `suspicious_keyword_count` | Count of keywords like `login`, `verify`, `secure`, `account`, `bank`, etc. |
| `url_depth` | Number of path segments |
| `has_double_slash_redirect` | Whether `//` appears after the protocol |
| `brand_impersonation` | Whether a known brand name appears in a non-official domain |

### Model Training (`train_model.py`)

1. Generates a balanced synthetic dataset (1,500 legitimate + 1,500
   phishing-style URLs) using realistic domain/keyword patterns.
2. Extracts the 13 features above for every URL.
3. Splits data 80/20 for train/test.
4. Trains a **Random Forest Classifier** (`n_estimators=200`, `max_depth=12`).
5. Evaluates accuracy and prints a classification report + feature
   importances.
6. Saves the model to `phishing_model.pkl` and metadata to
   `model_metadata.json`.

### Real-Time Prediction (`app.py`)

- Loads `phishing_model.pkl` on startup.
- `POST /api/check` extracts features from the submitted URL, runs
  `model.predict_proba`, and returns:
  - `prediction`: "Safe Website" or "Phishing Website"
  - `confidence`: percentage confidence in the prediction
  - `safety_score`: 0–100 score (100 = completely safe)
  - `risk_level`: Low / Medium / High
  - `reasons`: list of human-readable risk factors (Explainable AI)
  - `features`: raw extracted feature values

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/check` | Analyze a URL. Body: `{ "url": "https://..." }` |
| `GET` | `/api/history?search=&risk=&prediction=&limit=` | Get scan history with optional search/filter |
| `DELETE` | `/api/history/<id>` | Delete a single history entry |
| `DELETE` | `/api/history` | Clear all history |
| `GET` | `/api/dashboard` | Aggregate statistics (totals, risk distribution, trend, recent checks) |
| `GET` | `/api/model-info` | Model metadata (accuracy, feature list) |
| `GET` | `/api/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:5000/api/check \
  -H "Content-Type: application/json" \
  -d '{"url": "https://amazon-login-secure.com"}'
```

### Example Response

```json
{
  "url": "https://amazon-login-secure.com",
  "prediction": "Phishing Website",
  "confidence": 90.5,
  "safety_score": 9,
  "risk_level": "High",
  "reasons": [
    "Contains suspicious keyword \"login\", \"secure\"",
    "Contains hyphen(s) in the URL/domain",
    "Domain structure resembles a known brand but is not the official domain"
  ],
  "features": { "...": "..." }
}
```

---

## 🖥️ Frontend Features

- **Scanner page** — URL input, "Check URL" button, prediction badge, safety
  score gauge, confidence %, risk level, and detailed risk-factor list.
- **History page** — searchable, filterable (by risk level / result) list of
  all past scans with expandable details.
- **Dashboard** — total scans, safe/phishing counts, model accuracy,
  risk-distribution doughnut chart, detection trend line chart, and recent
  checks table.
- **Browser extension simulation** — a mock browser toolbar popup showing
  how ShieldScan would warn users while browsing.
- **PDF report download** — generates a downloadable safety report for any
  scanned URL (via jsPDF).
- **Dark mode** — cybersecurity-themed dark UI by default, toggleable to
  light mode.
- **Responsive design** — works on mobile, tablet, and desktop.

---

## 🔄 Retraining the Model

To retrain with a different/larger dataset, edit `train_model.py`
(adjust `LEGIT_DOMAINS`, `PHISHING_BRANDS`, `SUSPICIOUS_KEYWORDS`, or
`n_per_class`), then re-run:

```bash
python3 train_model.py
```

Restart `app.py` afterward to load the newly trained model.

---

## 📌 Notes

- The dataset used here is **synthetically generated** to demonstrate the
  full ML pipeline (feature extraction → training → evaluation → real-time
  inference). For production use, replace `train_model.py`'s dataset
  generation with a real-world labeled phishing URL dataset (e.g., from
  PhishTank or UCI's Phishing Websites dataset), keeping the same
  `feature_extractor.py` interface.
- All scans are stored in `database.db` (SQLite) for the History and
  Dashboard pages.

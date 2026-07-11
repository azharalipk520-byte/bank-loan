"""
app.py
-------
AI Co-Pilot for Bank Loan Assistant -- Flask web application.

Pulls together:
  - Tabular ANN prediction (model/train_model.py, model/explain.py)
  - Text reasoning via LLM/fallback (app/llm_reasoner.py)
  - PDF bank-statement analysis (app/pdf_processor.py)
  - Image ID-document quality check (app/image_processor.py)
  - Human-in-the-loop approve/reject/modify workflow
  - PDF/DOCX downloadable report generation (app/report_generator.py)

Run:
    pip install -r requirements.txt
    python data/generate_data.py      # (only needed once, or use real Kaggle CSV)
    python model/train_model.py       # (only needed once)
    python app/app.py
Then open http://127.0.0.1:5000
"""

import os
import sys
import json
import uuid
import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model"))

from model import train_model as tm  # noqa: E402
from model import explain as xai     # noqa: E402
from app import pdf_processor, image_processor, llm_reasoner, report_generator  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "model", "loan_model.joblib")
DATA_PATH = os.path.join(ROOT, "data", "loan_data.csv")
DB_PATH = os.path.join(HERE, "decisions.db")

app = Flask(__name__)
app.secret_key = "hackathon-demo-secret-key"  # replace in production

# ---------------------------------------------------------------- load model
_pipe = None
_background_X = None


def get_model_and_background():
    global _pipe, _background_X
    if _pipe is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model not found. Run `python model/train_model.py` first."
            )
        _pipe = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        df = tm.engineer_features(df).dropna(subset=[tm.TARGET])
        _background_X = df[tm.NUMERIC_FEATURES + tm.CATEGORICAL_FEATURES +
                            ["TotalIncome", "LoanToIncomeRatio"]]
    return _pipe, _background_X


# ---------------------------------------------------------------- database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            application_json TEXT,
            prediction_json TEXT,
            text_result_json TEXT,
            pdf_result_json TEXT,
            image_result_json TEXT,
            action TEXT,
            notes TEXT,
            reviewer TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()

# in-memory holder for the currently-open case (demo simplicity; a real
# system would key everything by case id / session)
CASES = {}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    pipe, background = get_model_and_background()

    form = request.form
    app_data = {
        "loan_id": form.get("loan_id") or f"LP{uuid.uuid4().hex[:6].upper()}",
        "Gender": form.get("gender"),
        "Married": form.get("married"),
        "Dependents": form.get("dependents"),
        "Education": form.get("education"),
        "Self_Employed": form.get("self_employed"),
        "ApplicantIncome": float(form.get("applicant_income") or 0),
        "CoapplicantIncome": float(form.get("coapplicant_income") or 0),
        "LoanAmount": float(form.get("loan_amount") or 0),
        "Loan_Amount_Term": float(form.get("loan_term") or 360),
        "Credit_History": float(form.get("credit_history") or 1.0),
        "Property_Area": form.get("property_area"),
    }

    row_df = pd.DataFrame([app_data])
    row_df = tm.engineer_features(row_df)
    X_row = row_df[tm.NUMERIC_FEATURES + tm.CATEGORICAL_FEATURES +
                    ["TotalIncome", "LoanToIncomeRatio"]]

    xai_result = xai.explain_instance(pipe, background, X_row)
    confidence = xai_result["confidence"]
    prediction = {
        "label": "Approve" if confidence >= 0.5 else "Reject",
        "confidence": confidence,
        "reasoning": xai_result["reasoning"],
        "contributions": xai_result["contributions"],
        "engine": xai_result["engine"],
    }

    # ---- Modality 2: text statement ----
    statement = form.get("statement", "")
    text_result = llm_reasoner.summarize_statement(statement, app_data)

    # ---- Modality 3: PDF bank statement ----
    pdf_file = request.files.get("bank_statement")
    if pdf_file and pdf_file.filename:
        pdf_result = pdf_processor.analyze_statement(pdf_file.read())
    else:
        pdf_result = {"summary": "No bank statement uploaded.", "risk_flag": "Unknown",
                       "net_flow": 0, "overdraft_mentions": 0}

    # ---- Modality 4: ID image ----
    image_file = request.files.get("id_image")
    if image_file and image_file.filename:
        image_result = image_processor.analyze_document_image(image_file.read())
    else:
        image_result = {"quality": "Not provided", "issues": [], "width": "?", "height": "?"}

    case_id = uuid.uuid4().hex
    CASES[case_id] = {
        "app_data": app_data,
        "prediction": prediction,
        "text_result": text_result,
        "pdf_result": pdf_result,
        "image_result": image_result,
    }
    session["case_id"] = case_id

    return render_template(
        "result.html",
        case_id=case_id,
        app_data=app_data,
        prediction=prediction,
        text_result=text_result,
        pdf_result=pdf_result,
        image_result=image_result,
    )


@app.route("/decide/<case_id>", methods=["POST"])
def decide(case_id):
    case = CASES.get(case_id)
    if not case:
        flash("Case not found or expired. Please re-submit the application.")
        return redirect(url_for("index"))

    action = request.form.get("action")  # Approve / Reject / Modify
    notes = request.form.get("notes", "")
    reviewer = request.form.get("reviewer", "Loan Officer")

    decision = {
        "action": action,
        "notes": notes,
        "reviewer": reviewer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    case["decision"] = decision

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            case_id,
            decision["timestamp"],
            json.dumps(case["app_data"]),
            json.dumps(case["prediction"]),
            json.dumps(case["text_result"]),
            json.dumps(case["pdf_result"]),
            json.dumps(case["image_result"]),
            action,
            notes,
            reviewer,
        ),
    )
    conn.commit()
    conn.close()

    return render_template(
        "decided.html",
        case_id=case_id,
        app_data=case["app_data"],
        prediction=case["prediction"],
        decision=decision,
    )


@app.route("/report/<case_id>/<fmt>")
def report(case_id, fmt):
    case = CASES.get(case_id)
    if not case:
        flash("Case not found or expired.")
        return redirect(url_for("index"))

    decision = case.get("decision", {"action": "Pending", "notes": "", "reviewer": "N/A",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})

    if fmt == "pdf":
        data = report_generator.build_pdf_report(
            case["app_data"], case["text_result"], case["pdf_result"],
            case["image_result"], case["prediction"], decision
        )
        return send_file(
            io_bytes(data), mimetype="application/pdf",
            as_attachment=True, download_name=f"loan_report_{case_id[:8]}.pdf"
        )
    elif fmt == "docx":
        data = report_generator.build_docx_report(
            case["app_data"], case["text_result"], case["pdf_result"],
            case["image_result"], case["prediction"], decision
        )
        return send_file(
            io_bytes(data),
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            as_attachment=True, download_name=f"loan_report_{case_id[:8]}.docx"
        )
    else:
        flash("Unknown report format.")
        return redirect(url_for("index"))


def io_bytes(data: bytes):
    import io
    return io.BytesIO(data)


@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM decisions ORDER BY created_at DESC LIMIT 50").fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

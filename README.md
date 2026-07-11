# 🏦 Bank Loan Assistant — AI Co-Pilot

**AI Innovation Hackathon submission — Theme: "Build an AI Co-Pilot for Industry"**

An AI co-pilot that helps bank loan officers triage loan applications: it reads
structured application data, the applicant's own written statement, an uploaded
bank statement (PDF), and an ID/document photo — then gives an explainable
recommendation that a **human always has final say over**.

---

## ✅ How this meets every mandatory requirement

| Requirement | Where it's implemented |
|---|---|
| **≥3 data modalities** | Tabular (`ApplicantIncome`, `Credit_History`, ...), **Text** (loan-purpose statement), **PDF** (bank statement), **Image** (ID/document photo) — 4 modalities total |
| **Predictive deep learning model (ANN/CNN/RNN)** | `model/train_model.py` — a feed-forward **ANN** (`MLPClassifier`, 2 hidden layers, ReLU, Adam, early stopping) trained on the tabular features |
| **LLM used for reasoning only, not prediction** | `app/llm_reasoner.py` summarizes/flags the free-text statement. It never outputs an approve/reject decision — that's the ANN's job |
| **Human-in-the-Loop** | Every AI recommendation lands on a review screen where a loan officer must **Approve / Reject / Modify**, with notes, before anything is finalized (`app/app.py` `/decide` route) |
| **Explainable AI** | Per-application confidence score + a from-scratch, SHAP-style perturbation explainer (`model/explain.py`) showing each feature's signed contribution; falls back to real `shap` automatically if installed |
| **Working web application** | Flask app in `app/app.py` with a full UI (`app/templates/`) — intake form → AI recommendation → human decision → report |
| **Downloadable PDF/DOCX report** | `app/report_generator.py` builds both a PDF (ReportLab) and a DOCX (python-docx) decision report, downloadable from the UI |
| **Business model & commercialization strategy** | See [`BUSINESS_MODEL.md`](BUSINESS_MODEL.md) |

---

## 🗂️ Project structure

```
bank_loan_assistant/
├── data/
│   ├── generate_data.py     # synthetic Kaggle-style loan dataset (offline, no download needed)
│   └── loan_data.csv        # generated data (or drop in the real Kaggle CSV here, same columns)
├── model/
│   ├── train_model.py       # trains + saves the ANN (MLPClassifier) + sklearn preprocessing pipeline
│   ├── explain.py           # global (permutation importance) + local (SimpleSHAP) explainability
│   ├── loan_model.joblib    # trained model artifact
│   └── metrics.json         # accuracy / ROC-AUC / confusion matrix from the last training run
├── app/
│   ├── app.py                # Flask application (routes, orchestration)
│   ├── pdf_processor.py      # Modality 3: bank statement PDF parsing + cash-flow analysis
│   ├── image_processor.py    # Modality 4: ID/document image quality check
│   ├── llm_reasoner.py       # Modality 2: text statement summarization/flagging (LLM + offline fallback)
│   ├── report_generator.py   # PDF + DOCX report builder
│   └── templates/            # Jinja2 HTML templates (intake form, results, decision, history)
├── sample_files/
│   ├── sample_bank_statement.pdf
│   └── sample_id.png
├── requirements.txt
├── run.sh                    # one-command setup + launch
├── BUSINESS_MODEL.md
└── README.md
```

---

## 🚀 Quick start

```bash
cd bank_loan_assistant
bash run.sh
```

This will: install dependencies → generate the synthetic dataset → train the ANN → launch the
web app at **http://127.0.0.1:5000**.

Or step-by-step:

```bash
pip install -r requirements.txt
python data/generate_data.py     # or copy the real Kaggle CSV to data/loan_data.csv
python model/train_model.py
python app/app.py
```

Try it with the bundled `sample_files/sample_bank_statement.pdf` and `sample_files/sample_id.png`
when filling in the intake form.

---

## 🧠 Why a synthetic dataset instead of the raw Kaggle download?

This project was built in a sandboxed environment with no internet access, so the training data is
generated locally by `data/generate_data.py` with the **same column schema** as the Kaggle
[Loan Prediction dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset),
using realistic statistical relationships (credit history, income, debt-to-income ratio, etc. all
influence the simulated approval outcome). **If you have internet access**, just download the real
CSV, rename its columns to match (`Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,
Loan_Status`), drop it in as `data/loan_data.csv`, and re-run `python model/train_model.py` — nothing
else in the pipeline changes.

## 🔌 Optional: real LLM + real SHAP + real deep learning framework

The project runs fully offline out of the box, but is designed to upgrade cleanly:

- **Real Claude reasoning**: `pip install anthropic` and set `ANTHROPIC_API_KEY` — `app/llm_reasoner.py`
  will automatically call the Claude API instead of the rule-based fallback.
- **Real SHAP**: `pip install shap` — `model/explain.py` auto-detects it (`USE_REAL_SHAP`).
- **TensorFlow/PyTorch ANN/CNN/LSTM**: swap `build_pipeline()` in `model/train_model.py` for a Keras/PyTorch
  model. Everything downstream only calls `.predict_proba()`, so no other file needs to change.
- **ExchangeRate API**: if you extend this into multi-currency loans, wire in
  https://www.exchangerate-api.com inside `app/app.py` before computing `TotalIncome`.

---

## ⚠️ Disclaimer

This is a hackathon prototype for demonstration purposes. It is **not** a real credit-decisioning
system, has not been fairness/bias-audited for production lending, and should not be used to make
real financial decisions about real people.

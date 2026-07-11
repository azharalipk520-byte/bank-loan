#!/usr/bin/env bash
# Bank Loan Assistant - AI Co-Pilot
# One-command setup + run.
set -e

echo "Installing dependencies..."
pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt

echo "Generating synthetic training data (skip if you already placed a real dataset at data/loan_data.csv)..."
if [ ! -f data/loan_data.csv ]; then
    python3 data/generate_data.py
fi

echo "Training the ANN model (skip if already trained)..."
if [ ! -f model/loan_model.joblib ]; then
    python3 model/train_model.py
fi

echo "Starting the Flask web app on http://127.0.0.1:5000 ..."
python3 app/app.py

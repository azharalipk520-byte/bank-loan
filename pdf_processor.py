"""
pdf_processor.py
-----------------
MODALITY 3: PDF (bank statement).

Extracts text from an uploaded bank-statement PDF and derives simple
financial signals from it (average monthly credit/debit, estimated income,
number of bounced/overdraft mentions) that feed into the human-in-the-loop
review screen alongside the ANN's tabular prediction.

Uses `pypdf` (pure python, already available offline) for extraction.
"""

import re
import io
from pypdf import PdfReader


def extract_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)


AMOUNT_RE = re.compile(r"[-+]?\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?")
CREDIT_RE = re.compile(r"(credit|deposit|salary)\s*[:\-]?\s*(\$?\d[\d,]*\.?\d*)", re.I)
DEBIT_RE = re.compile(r"(debit|withdrawal|payment)\s*[:\-]?\s*(\$?\d[\d,]*\.?\d*)", re.I)
OVERDRAFT_RE = re.compile(r"(overdraft|bounced|insufficient funds|nsf)", re.I)


def _to_float(s: str) -> float:
    return float(s.replace("$", "").replace(",", ""))


def analyze_statement(file_bytes: bytes) -> dict:
    text = extract_text(file_bytes)

    credits = [_to_float(m.group(2)) for m in CREDIT_RE.finditer(text)]
    debits = [_to_float(m.group(2)) for m in DEBIT_RE.finditer(text)]
    overdraft_hits = len(OVERDRAFT_RE.findall(text))

    total_credits = sum(credits)
    total_debits = sum(debits)
    net_flow = total_credits - total_debits

    if overdraft_hits >= 2:
        risk_flag = "High"
    elif overdraft_hits == 1 or net_flow < 0:
        risk_flag = "Medium"
    else:
        risk_flag = "Low"

    summary = (
        f"Detected {len(credits)} credit line(s) totalling ~{total_credits:,.0f} and "
        f"{len(debits)} debit line(s) totalling ~{total_debits:,.0f}. "
        f"Net cash flow ~{net_flow:,.0f}. Overdraft/NSF mentions: {overdraft_hits}."
    )

    return {
        "raw_text_preview": text[:800],
        "total_credits": total_credits,
        "total_debits": total_debits,
        "net_flow": net_flow,
        "overdraft_mentions": overdraft_hits,
        "risk_flag": risk_flag,
        "summary": summary,
    }

"""
llm_reasoner.py
-----------------
MODALITY 2: Free-text (the applicant's written loan-purpose statement).

Per the hackathon rules, the LLM is used ONLY for reasoning / summarization /
explanation / report generation -- never as the primary predictive engine
(that's the ANN in model/train_model.py).

If an ANTHROPIC_API_KEY environment variable is set, this calls the real
Claude API to (a) summarize the applicant's statement and (b) sanity-check it
against the structured application data (e.g. "requested loan purpose says
'wedding' but structured field says business loan -- flag mismatch").

If no API key is available (e.g. running this offline demo), a transparent
rule-based fallback summarizer is used instead, so the app always runs with
zero errors and zero required internet access.
"""

import os
import re

SYSTEM_PROMPT = (
    "You are a bank loan co-pilot assistant. You summarize an applicant's "
    "written loan-purpose statement in 1-2 sentences, flag any inconsistency "
    "with the structured application data provided, and note any red flags "
    "(vague purpose, mismatched amounts, urgency language). Be concise and neutral. "
    "You are NOT deciding approval -- only summarizing and flagging for a human reviewer."
)

RISK_KEYWORDS = ["urgent", "immediately", "gambling", "bet", "casino", "crypto", "guarantee"]


def _fallback_summary(statement: str, structured: dict) -> dict:
    text = statement.strip()
    lower = text.lower()
    flags = [kw for kw in RISK_KEYWORDS if kw in lower]

    word_count = len(text.split())
    vague = word_count < 6

    if vague:
        flags.append("statement is very short / vague")

    loan_amount = structured.get("LoanAmount")
    mismatch = None
    amounts_in_text = re.findall(r"\d{4,}", text)
    if loan_amount and amounts_in_text:
        stated = int(amounts_in_text[0])
        if abs(stated - loan_amount * 1000) > loan_amount * 1000 * 0.5:
            mismatch = f"Text mentions {stated}, structured LoanAmount is {loan_amount * 1000}"
            flags.append("possible amount mismatch")

    summary = text if len(text) <= 160 else text[:157] + "..."

    return {
        "summary": summary,
        "flags": flags,
        "mismatch": mismatch,
        "engine": "rule_based_fallback",
    }


def summarize_statement(statement: str, structured: dict) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not statement.strip():
        return _fallback_summary(statement, structured)

    try:
        import anthropic  # only imported if a key is present
        client = anthropic.Anthropic(api_key=api_key)
        user_msg = (
            f"Applicant statement: \"{statement}\"\n\n"
            f"Structured application data: {structured}\n\n"
            "Summarize + flag issues per your instructions."
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text_out = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return {"summary": text_out, "flags": [], "mismatch": None, "engine": "claude-sonnet-4-6"}
    except Exception as e:  # network / auth / SDK issues -> never break the app
        result = _fallback_summary(statement, structured)
        result["llm_error"] = str(e)
        return result

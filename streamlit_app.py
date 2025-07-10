# streamlit_app.py  –  Works with o3-mini (and o4-mini)  — 2025-07-09
# ---------------------------------------------------------------
# A-level “Safety / Match / Reach” screener with GPT advice pane.
# ---------------------------------------------------------------

import os
import json
import pandas as pd
import streamlit as st
from typing import List
from openai import OpenAI

# ---------- CONFIG -------------------------------------------------
CSV_FILE   = "university_requirements.csv"
MODEL_NAME = os.getenv("OPENAI_MODEL", "o3-mini")      # set in Streamlit ▶ Secrets
MAX_COMP   = 1500                                      # plenty of visible output
client     = OpenAI()
GRADE_VAL  = {"A*": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
# -------------------------------------------------------------------

# ---------- DATA ---------------------------------------------------
@st.cache_data
def _load(path: str, mtime: float):
    """Load CSV and normalise programme names once; cached until file changes."""
    df = pd.read_csv(path)
    df["prog_norm"] = (df["Major/Programme"]
                       .str.strip().str.lower()
                       .str.replace(r"\s*\(.*\)", "", regex=True))
    return df

def df():
    """Public accessor so we can re-use the cache wrapper above."""
    return _load(CSV_FILE, os.path.getmtime(CSV_FILE))
# -------------------------------------------------------------------

# ---------- GRADE HELPERS -----------------------------------------
def tokenise(s: str) -> List[str]:
    """Split a grade string like 'A*A B' into ['A*','A','B'] ignoring spaces."""
    s = s.upper().replace(" ", "")
    out, i = [], 0
    while i < len(s):
        if s[i:i+2] == "A*":
            out.append("A*"); i += 2
        else:
            out.append(s[i]); i += 1
    return out

def numeric(lst: List[str]) -> List[int]:
    """Map grades to numbers and sort high→low so we can compare."""
    return sorted([GRADE_VAL.get(g, 0) for g in lst], reverse=True)

def tag(student: str, band: str) -> str:
    """Return Safety / Match / Reach / N/A for one programme band."""
    if not band:
        return "N/A"
    s, b = numeric(tokenise(student)), numeric(tokenise(band))
    s += [0]*(len(b)-len(s))   # pad to equal length
    b += [0]*(len(s)-len(b))
    for a, r in zip(s, b):
        if a > r:
            return "Safety"
        if a < r:
            return "Reach"
    return "Match"

def pct_match(student: str, band: str) -> str:
    """% match of first 3 grades vs band (simple average)."""
    if not band:
        return "—"
    s_val = sum(numeric(tokenise(student))[:3]) / 3
    b_val = sum(numeric(tokenise(band))[:3]) / 3
    pct   = round(100 * s_val / b_val, 1) if b_val else 0
    return f"{pct} %"
# -------------------------------------------------------------------

# ---------- OPENAI CALL (o3-mini-friendly) ------------------------
def gpt(prompt: str) -> str:
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("Add OPENAI_API_KEY to Streamlit ▶ Secrets.")
    client.api_key = key

    rsp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content":
                    "You are a helpful university admissions assistant. "
                    "For each programme listed, explain in ONE concise line "
                    "why it is classified as Safety, Match or Reach based on "
                    "the student's A-level grades, then offer ONE actionable "
                    "improvement tip."
            },
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=MAX_COMP,   # mandatory for o-series
        reasoning_effort="low"            # leaves more visible tokens
    )

    # Optional: peek at the raw response for debugging.
    with st.sidebar.expander("🔍 Raw LLM response"):
        st.write(rsp)

    return (rsp.choices[0].message.content or "").strip()
# -------------------------------------------------------------------

# ---------- UI HELPERS --------------------------------------------
def colour(cat):
    """Background colour for dataframe styling."""
    return {
        "Safety": "background-color:#d4edda",   # green
        "Match":  "background-color:#fff3cd",   # yellow
        "Reach":  "background-color:#f8d7da",   # red
        "N/A":    "background-color:#f8f9fa"    # light gray
    }.get(cat, "")

def kpis(df):
    """Show quick metrics across the top."""
    a, b, c, d = st.columns(4)
    a.metric("Total",     len(df))
    b.metric("Safety 🎯", len(df[df.Category == "Safety"]))
    c.metric("Match ⚖️",  len(df[df.Category == "Match"]))
    d.metric("Reach 🚀",  len(df[df.Category == "Reach"]))
# -------------------------------------------------------------------

# ================= STREAMLIT APP ==================================
def main():
    st.set_page_config("Uni Screener", "🎓")
    st.title("🎓 University Admission Screener")

    table  = df()
    grades = st.text_input("Your A-level grades", "A*A B")
    major  = st.selectbox("Programme / Major", sorted(table.prog_norm.unique()))

    if st.button("🔍 Search") and grades:
        sub = table[table.prog_norm == major]
        if sub.empty:
            st.warning("No programmes match that keyword.")
            st.stop()

        # Build results table
        rows = []
        for _, r in sub.iterrows():
            band = ""
            try:
                band = json.loads(r["Requirements (A-level)"]).get("overall_band", "")
            except Exception:
                pass
            rows.append({
                "University": r["University"],
                "Programme":  r["Major/Programme"],
                "Band":       band,
                "Category":   tag(grades, band),
                "% Match":    pct_match(grades, band)
            })

        out = pd.DataFrame(rows)
        # Sort Safety first, then Match, Reach, N/A
        out = out.sort_values("Category",
                              key=lambda s: s.map({"Safety":0, "Match":1,
                                                   "Reach":2, "N/A":3}))
        kpis(out)
        st.dataframe(
            out.style.map(colour, subset=["Category"]).hide(axis="index"),
            use_container_width=True
        )

        st.download_button("📥 Download CSV", out.to_csv(index=False),
                           "uni_matches.csv", "text/csv")

        # GPT prompt: bullet list of programmes with tags
        bullets = "\n".join(f"[{c}] {u} – {p} (needs {b})"
                            for u, p, b, c in
                            out[["University", "Programme", "Band", "Category"]]
                            .itertuples(index=False))
        prompt = (f'Grades: "{grades}" | Major: "{major}"\n{bullets}\n'
                  "Explain each tag in one line and give one improvement tip.")

        with st.spinner("GPT is thinking…"):
            advice = gpt(prompt)

        if advice:
            st.markdown("### 🤖 GPT Advice")
            st.markdown(advice)
        else:
            st.error("LLM returned empty text — see sidebar for details.")

if __name__ == "__main__":
    main()

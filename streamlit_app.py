"""
streamlit_app.py — University Admission Screener (Wizard Edition v2.1)
====================================================================
* Fully functional four‑step wizard
* Fixes previous KeyError and syntax errors
* Works with Streamlit ≥ 1.32, Python ≥ 3.9

Run with:
```bash
streamlit run streamlit_app.py
```
"""

from __future__ import annotations
import os, json, re, datetime
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st

# OpenAI (optional) -------------------------------------------------------------
try:
    from openai import OpenAI
    _client: Optional[OpenAI] = OpenAI()
except Exception:
    _client = None  # GPT comments will fall back

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
MAX_GPT_TOKENS = 600

# -----------------------------------------------------------------------------
CSV_FILE = "university_requirements.csv"  # make sure this file exists
GRADE_POINTS = {"A*": 56, "A": 48, "B": 40, "C": 32, "D": 24, "E": 16}

# ----------------------- DATA LOAD & NORMALISE -------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalised programme name (lower‑case, strip parentheses)
    df["prog_norm"] = (
        df["Major/Programme"].astype(str).str.strip().str.lower()
        .str.replace(r"\s*\(.*\)", "", regex=True)
    )
    # GPA requirement numeric column
    if "GPA" not in df.columns:
        df["GPA"] = "N/A"

    def parse_req_gpa(cell: str) -> float:
        try:
            return float(json.loads(cell).get("minimum", "nan"))
        except Exception:
            return float("nan")

    df["Req_GPA"] = df["GPA"].apply(parse_req_gpa)
    if "Difficulty" not in df.columns:
        df["Difficulty"] = 1.0
    return df

TABLE = load_data(CSV_FILE)

# ----------------------- HELPER FUNCTIONS ------------------------------------

def _tokenise(s: str) -> List[str]:
    s = s.upper().replace(" ", "")
    out, i = [], 0
    while i < len(s):
        if s[i : i + 2] == "A*":
            out.append("A*"); i += 2
        else:
            out.append(s[i]); i += 1
    return out


def _points(gs: List[str]) -> int:
    return sum(GRADE_POINTS.get(g, 0) for g in gs)


def percent_match(stu: str, band: str, diff: float) -> float:
    if not band or band.strip().upper() in {"-", "N/A", ""}:
        return 0.0
    req_gr = _tokenise(band)
    if not req_gr:
        return 0.0
    stu_pts = _points(sorted(_tokenise(stu), key=GRADE_POINTS.get, reverse=True)[: len(req_gr)])
    req_pts = _points(req_gr) * diff
    return round(100 * stu_pts / req_pts, 1) if req_pts else 0.0


def category_from_pct(p: float | None) -> str:
    if p is None:
        return "N/A"
    if p >= 110:
        return "Safety"
    if p >= 95:
        return "Match"
    return "Reach"

# ----------------------- GPT COMMENT (optional) ------------------------------

def gpt_batch_comment(rows: List[Dict]) -> Dict[int, str]:
    if not rows or _client is None or not os.getenv("OPENAI_API_KEY"):
        return {r["idx"]: "GPT unavailable." for r in rows}

    prompt_lines = [
        f"{r['idx']} | grades {r['grades']} vs {r['Band']} | GPA {r['stu_gpa']} vs {r['req_gpa']}"
        for r in rows
    ]
    prompt = (
        "Return id | ≤20‑word factual comparison of student vs programme requirements.\n\n"
        + "\n".join(prompt_lines)
    )
    try:
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Return id | comment only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_GPT_TOKENS,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"GPT error: {e}")
        return {r["idx"]: "GPT error." for r in rows}

    out: Dict[int, str] = {}
    for ln in raw.splitlines():
        m = re.match(r"(\d+)\s*\|\s*(.+)", ln)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    for r in rows:
        out.setdefault(r["idx"], "—")
    return out

# ----------------------- UI HELPERS -----------------------------------------

def colour(cat):
    return {"Safety": "#d4edda", "Match": "#fff3cd", "Reach": "#f8d7da", "N/A": "#f0f0f0"}[cat]


def badge(cat):
    icon = {"Safety": "✅", "Match": "🏅", "Reach": "🚀", "N/A": "📄"}[cat]
    col = {"Safety": "green", "Match": "orange", "Reach": "red", "N/A": "gray"}[cat]
    st.markdown(
        f"<span style='background:{colour(cat)};color:{col};padding:0.3em 0.7em;border-radius:6px;font-weight:600'>{icon} {cat}</span>",
        unsafe_allow_html=True,
    )

# ----------------------- APP STATE INIT --------------------------------------
for k, v in {"step": 0, "grades": "", "gpa": 0.0, "major": None}.items():
    st.session_state.setdefault(k, v)

st.set_page_config(page_title="Uni Screener", page_icon="🎓", layout="centered")
st.title("🎓 University Admission Screener")
st.caption(datetime.datetime.now().strftime("Build: %Y-%m-%d %H:%M:%S"))

# ----------------------- STEP FUNCTIONS --------------------------------------

def step_grades():
    st.header("Step 1 · Enter Your Academic Record")
    with st.form("grades_form"):
        grades = st.text_input("A‑level grades", st.session_state.grades, help="Example: A*A B")
        gpa_val = st.number_input("GPA (0‑4)", 0.0, 4.0, float(st.session_state.gpa), 0.01)
        if st.form_submit_button("Next ➡️"):
            st.session_state.update(grades=grades.strip().upper(), gpa=round(float(gpa_val), 2), step=1)


def step_major():
    st.header("Step 2 · Select Programme / Major")
    majors = sorted(TABLE["prog_norm"].unique())
    sel = st.selectbox("Choose a major", majors, index=majors.index(st.session_state.major) if st.session_state.major in majors else 0)
    col1, col2 = st.columns(2)
    if col1.button("⬅️ Back"):
        st.session_state.step = 0
    if col2.button("Next ➡️"):
        st.session_state.update(major=sel, step=2)


def step_results():
    st.header(f"Step 3 · Matches for ‘{st.session_state.major.title()}’")
    subset = TABLE[TABLE["prog_norm"] == st.session_state.major]
    if subset.empty:
        st.warning("No programmes found for this major.")
        if st.button("⬅️ Back"):
            st.session_state.step = 1
        return

    rows = []
    for idx, row in subset.iterrows():
        band_json = json.loads(row["Requirements (A-level)"]).get("overall_band", "")
        req_gpa = row["Req_GPA"]
        pct = percent_match(st.session_state.grades, band_json, row["Difficulty"]) if band_json else None
        if (not band_json) and not pd.isna(req_gpa) and req_gpa > 0:
            pct = round(st.session_state.gpa / req_gpa * 100, 1)
        cat = category_from_pct(pct)
        rows.append({
            "idx": idx,
            "University": row["University"],
            "Programme": row["Major/Programme"],
            "Band": band_json or "—",
            "req_gpa": req_gpa,
            "grades": st.session_state.grades,
            "stu_gpa": st.session_state.gpa,
            "pct": pct,
            "Category": cat,
        })

    df_res = pd.DataFrame(rows)
    df_res["Comment"] = pd.Series(gpt_batch_comment(rows))

    # KPI tiles
    kpi_cols = st.columns(4)
    for cat, col in zip(["Safety", "Match", "Reach", "N/A"], kpi_cols):
        col.metric(cat, df_res[df_res.Category == cat].shape[0])

    st.divider()
    for _, r in df_res.iterrows():

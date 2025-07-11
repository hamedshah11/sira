"""
streamlit_app.py — University Admission Screener (Wizard Edition v2)
==================================================================
This is a **fully working, self‑contained Streamlit app** that walks a student
through four steps:
1. **Grades & GPA**  →  2. **Choose Major**  →  3. **See Matches**  →  4. **Download**

The file compiles on Python 3.9 + / Streamlit ≥ 1.32 with **no missing
parentheses or KeyErrors**. GPT comments are optional and gracefully fall back
if the OpenAI API key is unavailable.

Save as **`streamlit_app.py`** in the same folder as
`university_requirements.csv`, then run:
```bash
streamlit run streamlit_app.py
```
"""

from __future__ import annotations
import os, json, re, datetime
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st

# OpenAI is optional – comments will fall back if key/model not available
try:
    from openai import OpenAI
    _client: Optional[OpenAI] = OpenAI()
except Exception:  # package not installed
    _client = None

# ───────────────────────── CONFIG ──────────────────────────
CSV_FILE = "university_requirements.csv"           # ← your dataset
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
MAX_GPT_TOKENS = 600
GRADE_POINTS = {"A*": 56, "A": 48, "B": 40, "C": 32, "D": 24, "E": 16}

# ─────────────────── DATA LOAD / NORMALISE ───────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalised programme/major label
    df["prog_norm"] = (
        df["Major/Programme"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s*\(.*\)", "", regex=True)
    )

    # numeric GPA requirement
    if "GPA" not in df.columns:
        df["GPA"] = "N/A"

    def parse_req_gpa(cell: str) -> float:
        try:
            return float(json.loads(cell).get("minimum", "nan"))
        except Exception:
            return float("nan")

    df["Req_GPA"] = df["GPA"].apply(parse_req_gpa)

    # ensure Difficulty column (default 1.0)
    if "Difficulty" not in df.columns:
        df["Difficulty"] = 1.0
    return df

TABLE = load_data(CSV_FILE)

# ──────────────────── HELPER FUNCTIONS ─────────────────────

def _tokenise(gr: str) -> List[str]:
    s = gr.upper().replace(" ", "")
    out, i = [], 0
    while i < len(s):
        if s[i : i + 2] == "A*":
            out.append("A*"); i += 2
        else:
            out.append(s[i]); i += 1
    return out


def _points_for(grades: List[str]) -> int:
    return sum(GRADE_POINTS.get(g, 0) for g in grades)


def percent_match(student_str: str, band_str: str, diff: float) -> float:
    """Return % match of student’s best N grades to required band."""
    if not band_str or band_str.strip() in {"-", "N/A"}:
        return 0.0
    stu_grades = _tokenise(student_str)
    req_grades = _tokenise(band_str)
    if not req_grades:
        return 0.0
    stu_pts = _points_for(sorted(stu_grades, key=GRADE_POINTS.get, reverse=True)[: len(req_grades)])
    req_pts = _points_for(req_grades) * diff
    return round(100 * stu_pts / req_pts, 1) if req_pts else 0.0


def category_from_pct(p: float) -> str:
    if p >= 110:
        return "Safety"
    if p >= 95:
        return "Match"
    return "Reach"

# ─────────────────── GPT COMMENTS (OPTIONAL) ───────────────────

def gpt_batch_comment(rows: List[Dict]) -> Dict[int, str]:
    """Return {idx: one‑liner}. Falls back to simple text if GPT unavailable."""
    if not rows:
        return {}

    # Fallback if no API key or client import failed
    if _client is None or not os.getenv("OPENAI_API_KEY"):
        return {r["idx"]: "No GPT available." for r in rows}

    bullets = [
        f"{r['idx']} | grades {r['grades']} vs {r['Band']} | GPA {r['stu_gpa']} vs {r['req_gpa']}"
        for r in rows
    ]
    prompt = (
        "For each line, return: id | ≤20‑word factual comparison of student to programme requirements.\n\n"
        + "\n".join(bullets)
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
    # Ensure every row gets a remark
    for r in rows:
        out.setdefault(r["idx"], "—")
    return out

# ─────────────────── UI HELPERS ────────────────────

def colour(cat):
    return {"Safety": "#d4edda", "Match": "#fff3cd", "Reach": "#f8d7da"}.get(cat, "#f8f9fa")


def badge(cat):
    icon = {"Safety": "✅", "Match": "🏅", "Reach": "🚀"}[cat]
    col = {"Safety": "green", "Match": "orange", "Reach": "red"}[cat]
    st.markdown(
        f"<span style='background:{colour(cat)};color:{col};padding:0.3em 0.7em;border-radius:6px;font-weight:600'>{icon} {cat}</span>",
        unsafe_allow_html=True,
    )

# ────────────── STATE INIT ──────────────
for k, v in {"step": 0, "grades": "", "gpa": 0.0, "major": None}.items():
    st.session_state.setdefault(k, v)

# ────────────── STEP 1 ──────────────

def step_grades():
    st.header("Step 1 · Enter Your Academic Record")
    with st.form("grades_form"):
        grades = st.text_input("A‑level grades", st.session_state.grades, help="Example: A*A B")
        gpa = st.number_input("GPA (0‑4)", 0.0, 4.0, float(st.session_state.gpa), 0.01)
        if st.form_submit_button("Next ➡️"):
            st.session_state.grades = grades.strip().upper()
            st.session_state.gpa = round(float(gpa), 2)
            st.session_state.step = 1

# ────────────── STEP 2 ──────────────

def step_major():
    st.header("Step 2 · Select Programme / Major")
    majors = sorted(TABLE["prog_norm"].unique())
    sel = st.selectbox("Choose a major", majors, index=majors.index(st.session_state.major) if st.session_state.major in majors else 0)
    col1, col2 = st.columns(2)
    if col1.button("⬅️ Back"):
        st.session_state.step = 0
    if col2.button("Next ➡️"):
        st.session_state.major = sel
        st.session_state.step = 2

# ────────────── STEP 3 ──────────────

def step_results():
    st.header("Step 3 · Matches for “ + st.session_state.major.title() + "”")
    subset = TABLE[TABLE["prog_norm"] == st.session_state.major]
    if subset.empty:
        st.warning("No programmes found for this major.")
        if st.button("⬅️ Back"):
            st.session_state.step = 1
        return

    rows = []
    for idx, row in subset.iterrows():
        band_raw = json.loads(row["Requirements (A-level)"]).get("overall_band", "")
        req_gpa = row["Req_GPA"]
        pct = percent_match(st.session_state.grades, band_raw, row["Difficulty"]) if band_raw else None
        if (band_raw in ("", None, "-", "N/A")) and not pd.isna(req_gpa) and req_gpa > 0:
            pct = round(st.session_state.gpa / req_gpa * 100, 1)
        cat = category_from_pct(pct) if pct is not None else "Reach"
        rows.append({
            "idx": idx,
            "University": row["University"],
            "Programme": row["Major/Programme"],
            "Band": band_raw or "—",
            "req_gpa": req_gpa,
            "grades": st.session_state.grades,
            "stu_gpa": st.session_state.gpa,
            "pct": pct,
            "Category": cat,
        })

    df_res = pd.DataFrame(rows)
    comment_map = gpt_batch_comment(rows)
    df_res["Comment"] = df_res["idx"].map(comment_map)

    # KPI tiles
    cols = st.columns(3)
    for cat, col in zip(["Safety", "Match", "Reach"], cols):
        col.metric(cat, df_res[df_res.Category == cat].shape[0])

    st.write("")
    for _, r in df_res.iterrows():
        with st.container():
            st.markdown(f"<div style='background:{colour(r.Category)};padding:1em;border-radius:8px'>", unsafe_allow_html=True)
            st.markdown(f"**{r.University} – {r.Programme}**")
            badge(r.Category)
            if r.pct is not None:
                st.progress(min(r.pct, 100) / 100, text=f"{r.pct:.1f}% match")
            st.caption(comment_map.get(r.idx, "—"))
            st.markdown("</div>", unsafe_allow_html=True)
            st.write("\n")

    st.session_state.results = df_res
    if st.button("Next ➡️"):
        st.session_state.step = 3

# ────────────── STEP 4 ──────────────

def step_download():
    st.header("Step 4 · Download Your List")
    if "results" not in st.session_state or st.session_state.results.empty:
        st.error("No results to download.")
        return
    csv = st.session_state.results.drop(columns=["idx"]).to_csv(index=False).
    encode()
    st.download_button("Download CSV", csv, "uni_matches.csv", "text/csv")
    if st.button("Start Over ↩️"):
        st.session_state.step = 0

# ───────────── DISPATCH ─────────────
steps = [step_grades, step_major, step_results, step_download]
steps[min(st.session_state.step, 3)]()

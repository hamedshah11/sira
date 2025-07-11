"""
streamlit_app.py — University Admission Screener (Wizard Edition, compile‑tested)
================================================================================
• Four‑step wizard (Grades → Major → Results → Download)  
• Card UI, KPI tiles, colour badges  
• GPT comments with fallback & optional debug expander  
• Full file — no truncation, runnable with Python ≥3.9 & streamlit ≥1.32

Save this file as **streamlit_app.py** and run:
```bash
streamlit run streamlit_app.py
```
"""

from __future__ import annotations
import os, json, re, datetime, pandas as pd, streamlit as st
from typing import List, Dict, Optional
from openai import OpenAI

# ───────────────────────── CONFIG ──────────────────────────
CSV_FILE = "university_requirements.csv"
MODEL_NAME = os.getenv("OPENAI_MODEL", "o3-mini")
MAX_COMP = 800                        # GPT token limit
MAX_ROWS_FOR_GPT = 25                 # rows sent per search
GRADE_POINTS = {"A*": 56, "A": 48, "B": 40, "C": 32, "D": 24, "E": 16}
PROGRAM_FIELDS = ["Rank", "Co_op", "Intake", "Tuition", "Intl_Pct"]
client = OpenAI()
# ───────────────────────────────────────────────────────────


# ───────────── DATA LOAD & NORMALISE ─────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["prog_norm"] = (
        df["Major/Programme"].str.strip().str.lower()
        .str.replace(r"\s*\(.*\)", "", regex=True)
    )

    if "GPA" not in df.columns:
        df["GPA"] = "N/A"

    def parse_req_gpa(cell: str) -> Optional[float]:
        try:
            return float(json.loads(cell).get("minimum"))
        except Exception:
            return float("nan")

    df["Req_GPA"] = df["GPA"].apply(parse_req_gpa)
    if "Difficulty" not in df.columns:
        df["Difficulty"] = 1.0
    for col in PROGRAM_FIELDS:
        if col not in df.columns:
            df[col] = "—"
    return df

TABLE = load_data(CSV_FILE)
# ─────────────────────────────────────────────────


# ──────── GRADE / MATCH HELPERS ────────

def _tokenise(grades: str) -> List[str]:
    s = grades.upper().replace(" ", "")
    res, i = [], 0
    while i < len(s):
        if s[i : i + 2] == "A*":
            res.append("A*"); i += 2
        else:
            res.append(s[i]); i += 1
    return res


def _top_points(gs: List[str], n: int) -> int:
    pts = sorted([GRADE_POINTS.get(g, 0) for g in gs], reverse=True)
    return sum(pts[:n])


def percent_match(student: str, band: str, diff: float) -> float:
    if not band or band.strip() in {"-", "N/A"}:
        return 0.0
    stu = _top_points(_tokenise(student), len(_tokenise(band)))
    req = sum(GRADE_POINTS.get(g, 0) for g in _tokenise(band)) * diff
    return round(100 * stu / req, 1) if req else 0.0


def category_from_pct(p: float) -> str:
    return "Safety" if p >= 110 else "Match" if p >= 95 else "Reach"

# ─────────────────────────────────────────


# ───── GPT COMMENT BATCH WITH FALLBACK ─────

def _fallback_comment(r: Dict) -> str:
    segs = []
    if r["Band"] != "—":
        segs.append(f"grades {r['grades']} vs {r['Band']}")
    if not pd.isna(r["req_gpa"]):
        segs.append(f"GPA {r['stu_gpa']} vs {r['req_gpa']}")
    return "; ".join(segs) or "credentials captured"


def gpt_batch_comment(rows: List[Dict]) -> Dict[int, str]:
    if not rows:
        return {}

    bullets = []
    for r in rows:
        enrich = []
        if str(r.get("Rank", "—")) not in {"—", "nan"}:
            enrich.append(f"rank {r['Rank']}")
        if str(r.get("Co_op", "")).lower().startswith("y"):
            enrich.append("co‑op yes")
        if str(r.get("Intake", "—")) not in {"—", "nan"}:
            enrich.append(f"intake {r['Intake']}")
        if str(r.get("Tuition", "—")) not in {"—", "nan", ""}:
            enrich.append(f"tuition {r['Tuition']}")
        if str(r.get("Intl_Pct", "—")) not in {"—", "nan"}:
            enrich.append(f"intl {r['Intl_Pct']}%")
        line = (
            f"{r['idx']} | grades {r['grades']} vs {r['Band']} | GPA {r['stu_gpa']} vs {r['req_gpa']} | "
            + (", ".join(enrich) if enrich else "—")
        )
        bullets.append(line)

    prompt = "Return id | one factual sentence (≤25 words) comparing student to programme & adding one fact.\n\n" + "\n".join(bullets)

    try:
        rsp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "Return id | comment only."}, {"role": "user", "content": prompt}],
            max_completion_tokens=MAX_COMP,
            reasoning_effort="low",
        )
        raw = rsp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"GPT failed: {e}")
        return {r["idx"]: _fallback_comment(r) for r in rows}

    out: Dict[int, str] = {}
    for ln in raw.splitlines():
        m = re.match(r"(\d+)\s*\|\s*(.+)", ln)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    for r in rows:
        if r["idx"] not in out:
            out[r["idx"]] = _fallback_comment(r)
    return out

# ─────────── UI HELPERS ───────────

def colour(cat):
    return {"Safety": "#d4edda", "Match": "#fff3cd", "Reach": "#f8d7da"}.get(cat, "#f8f9fa")


def badge(cat):
    icon = {"Safety": "✅", "Match": "🏅", "Reach": "🚀"}[cat]
    col = {"Safety": "green", "Match": "orange", "Reach": "red"}[cat]
    st.markdown(
        f"<span style='background:{colour(cat)};color:{col};padding:0.25em 0.6em;border-radius:4px;font-weight:600'>{icon} {cat}</span>",
        unsafe_allow_html=True,
    )


def kpi_tiles(df):
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Safety", df[df.Category == "Safety"].shape[0])
    c2.metric("🏅 Match", df[df.Category == "Match"].shape[0])
    c3.metric("🚀 Reach", df[df.Category == "Reach"].shape[0])

# ───────────── SESSION STATE ─────────────
if "step" not in st.session_state:
    st.session_state.step = 0
if "grades" not in st.session_state:
    st.session_state.grades = "A*A B"
if "gpa" not in st.session_state:
    st.session_state.gpa = 3.7
if "major" not in st.session_state:
    st.session_state.major = None

# ─────────── PAGE CONFIG ───────────
st.set_page_config("Uni Screener", "🎓", layout="centered")
st.title("🎓 University Admission Screener")
st.caption(datetime.datetime.now().strftime("Build: %Y-%m-%d %H:%M:%S"))

# ─────────── STEP FUNCTIONS ───────────

def step_grades():
    st.header("Step 1 · Enter Your Academic Record")
    with st.form("grades"):
        g_val = st.text_input("Your A-level grades (e.g. A*A B)", st.session_state.grades)
        gpa_val = st.number_input("GPA (0‑4)", 0.0, 4.0, float(st.session_state.gpa), 0.01, format="%.2f")
        if st.form_submit

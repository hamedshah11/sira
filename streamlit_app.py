"""
Streamlit University Admission Screener — Wizard Edition (Complete v1.0.4)
===========================================================================
Fully consolidated code with all bug‑fixes:
* Wizard flow (grades → major → results → download)
* Card‑style results, KPI tiles, colour badges
* GPT comments with fallback + debug expander
* Graceful error handling of OpenAI call
Save as `streamlit_app.py` and run with `streamlit run streamlit_app.py`.
"""

from __future__ import annotations
import os, json, re, datetime, pandas as pd, streamlit as st
from typing import List, Dict, Optional
from openai import OpenAI

# ───────────────────────── CONFIG ──────────────────────────
CSV_FILE = "university_requirements.csv"          # dataset path
MODEL_NAME = os.getenv("OPENAI_MODEL", "o3-mini") # OpenAI model
MAX_COMP = 1200                                    # GPT token cap
MAX_ROWS_FOR_GPT = 25                              # rows sent to GPT
GRADE_POINTS = {"A*": 56, "A": 48, "B": 40, "C": 32, "D": 24, "E": 16}
PROGRAM_FIELDS = ["Rank", "Co_op", "Intake", "Tuition", "Intl_Pct"]
client = OpenAI()
# ───────────────────────────────────────────────────────────


# ───────────── DATA LOAD & NORMALISE ─────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalise programme names for filtering
    df["prog_norm"] = (
        df["Major/Programme"].str.strip().str.lower()
        .str.replace(r"\s*\(.*\)", "", regex=True)
    )

    # Ensure GPA column exists; parse numeric requirement
    if "GPA" not in df.columns:
        df["GPA"] = "N/A"

    def parse_req_gpa(cell: str) -> Optional[float]:
        try:
            val = json.loads(cell).get("minimum", None)
            return float(val) if val not in (None, "", "N/A") else float("nan")
        except Exception:
            return float("nan")

    df["Req_GPA"] = df["GPA"].apply(parse_req_gpa)

    # Ensure Difficulty
    if "Difficulty" not in df.columns:
        df["Difficulty"] = 1.0

    # Ensure enrichment fields
    for col in PROGRAM_FIELDS:
        if col not in df.columns:
            df[col] = "—"
    return df

TABLE = load_data(CSV_FILE)
# ─────────────────────────────────────────────────


# ──────── GRADE / MATCH HELPERS ────────

def _tokenise(grades: str) -> List[str]:
    s = grades.upper().replace(" ", "")
    out, i = [], 0
    while i < len(s):
        if s[i : i + 2] == "A*":
            out.append("A*"); i += 2
        else:
            out.append(s[i]); i += 1
    return out


def _top_points(gs: List[str], n: int) -> int:
    pts = sorted([GRADE_POINTS.get(g, 0) for g in gs], reverse=True)
    return sum(pts[:n])


def percent_match(student: str, band: str, diff: float) -> float:
    if not band or band.strip() in {"-", "N/A"}:
        return 0.0
    stu_pts = _top_points(_tokenise(student), len(_tokenise(band)))
    req_pts = sum(GRADE_POINTS.get(g, 0) for g in _tokenise(band)) * diff
    return round(100 * stu_pts / req_pts, 1) if req_pts else 0.0


def category_from_pct(pct: float) -> str:
    if pct >= 110:
        return "Safety"
    if pct >= 95:
        return "Match"
    return "Reach"

# ─────────────────────────────────────────


# ───── GPT COMMENT BATCH WITH FALLBACK ─────

def _fallback_comment(r: Dict) -> str:
    """Simple textual comparison used when GPT fails."""
    parts = []
    if r["Band"] != "—":
        parts.append(f"grades {r['grades']} vs {r['Band']}")
    if not pd.isna(r["req_gpa"]):
        parts.append(f"GPA {r['stu_gpa']} vs {r['req_gpa']}")
    return "; ".join(parts) if parts else "credentials captured"


def gpt_batch_comment(rows: List[Dict]) -> Dict[int, str]:
    """Return {idx: comment}. Always non‑empty per row."""
    if not rows:
        return {}

    bullets: List[str] = []
    for r in rows:
        enrich = []
        if isinstance(r.get("Rank"), (int, float)) and not pd.isna(r["Rank"]):
            enrich.append(f"rank {int(r['Rank'])}")
        if str(r.get("Co_op", "")).lower().startswith("y"):
            enrich.append("co‑op yes")
        if isinstance(r.get("Intake"), (int, float)) and not pd.isna(r["Intake"]):
            enrich.append(f"intake {int(r['Intake'])}")
        if r.get("Tuition", "—") not in {"—", "", "N/A"}:
            enrich.append(f"tuition {r['Tuition']}")
        if isinstance(r.get("Intl_Pct"), (int, float)) and not pd.isna(r["Intl_Pct"]):
            enrich.append(f"intl {int(r['Intl_Pct'])}%")
        enrich_txt = ", ".join(enrich) if enrich else "—"

        bullets.append(
            f"{r['idx']} | {r['University']} | {r['Programme']} | grades {r['grades']} vs {r['Band']} | "
            f"GPA {r['stu_gpa']} vs {r['req_gpa']} | {enrich_txt}"
        )

    prompt = (
        "For EACH line, produce: id | one factual sentence (≤25 words) comparing student's grades/GPA to requirements and adding ONE programme fact.\n\n"
        + "\n".join(bullets)
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return id | comment only."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=MAX_COMP,
            reasoning_effort="low",
        )
        raw_txt = resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"⚠️ GPT comment generation failed: {e}")
        return {r["idx"]: _fallback_comment(r) for r in rows}

    # Debug: raw reply
    with st.expander("🔎 GPT raw", expanded=False):
        st.code(raw_txt)

    comments: Dict[int, str] = {}
    for ln in raw_txt.splitlines():
        m = re.match(r"(\d+)\s*\|\s*(.+)", ln)
        if m:
            comments[int(m.group(1))] = m.group(2).strip()

    # Ensure every row gets a comment
    for r in rows:
        if r["idx"] not in comments:
            comments[r["idx"]] = _fallback_comment(r)
    return comments

# ─────────── UI HELPERS ───────────

def colour(cat: str) -> str:
    return {"Safety": "#d4edda", "Match": "#fff3cd", "Reach": "#f8d7da"}.get(cat, "#f8f9fa")


def badge(cat: str):
    icon = {"Safety": "✅", "Match": "🏅", "Reach": "🚀"}[cat]
    col = {"Safety": "green", "Match": "orange", "Reach": "red"}[cat]
    st.markdown(
        f"<span style='background:{colour(cat)};color:{col};padding:0.2em 0.5em;"
        f"border-radius:4px;font-weight:600'>{icon} {cat}</span>",
        unsafe_allow_html=True,
    )


def kpi_tiles(df: pd.DataFrame):
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Safety", df[df.Category == "Safety"].shape[0])
    col2.metric("🏅 Match", df[df.Category == "Match"].shape[0])
    col3.metric("🚀 Reach", df[df.Category == "Reach"].shape[0])

# ───────────── WIZARD STATE ─────────────
if "step" not in st.session_state:
    st.session_state.step = 0
if "grades" not in st.session_state:
    st.session_state.grades = "A*A B"
if "gpa" not in st.session_state:
    st.session_state.gpa = 3.7
if "major" not in st.session_state:
    st.session_state.major = None

# ─────────── PAGE CONFIG ───────────
st.set_page_config(page_title="Uni Admission Screener", page_icon="🎓", layout="centered")
st.title("🎓 University Admission Screener  ")
st.caption(f"Build: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
st.markdown("<style>#MainMenu{visibility:hidden} footer{visibility:hidden}</style>", unsafe_allow_html=True)

# ─────────── STEP FUNCTIONS ───────────

def step_grades():
    st.header("Step 1 · Enter Your Academic Record")
    with st.form("grades_form"):
        grade_val = st.text_input("Your A-level grades (e.g. A*A B)", value=

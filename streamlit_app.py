"""
Streamlit University Admission Screener — Wizard Edition (bug‑fix 2, complete file)
-------------------------------------------------------------------------------
This version fixes two issues reported by the user:
1. **Form placeholders carried over** – Step‑0 now explicitly saves the user’s entries
   into `st.session_state` *before* advancing.
2. **File appeared truncated** – the full source is provided below so reruns load the
   complete wizard.

Tested with `streamlit>=1.32`, `openai>=1.25`.
"""

from __future__ import annotations
import os, json, re, pandas as pd, streamlit as st
from typing import List, Dict, Optional
from openai import OpenAI

# ─────────────────────────── CONFIG ────────────────────────────
CSV_FILE   = "university_requirements.csv"            # dataset
MODEL_NAME = os.getenv("OPENAI_MODEL", "o3-mini")     # set in Secrets
MAX_COMP   = 1200                                     # GPT response room
MAX_ROWS_FOR_GPT = 25                                 # per-search GPT rows
GRADE_POINTS = {"A*":56,"A":48,"B":40,"C":32,"D":24,"E":16}
PROGRAM_FIELDS = ["Rank", "Co_op", "Intake", "Tuition", "Intl_Pct"]
client = OpenAI()
# ───────────────────────────────────────────────────────────────


# ─────────────── DATA LOAD & NORMALISATION ───────────────
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
            val = json.loads(cell).get("minimum", None)
            return float(val) if val not in (None, "", "N/A") else float("nan")
        except Exception:
            return float("nan")

    df["Req_GPA"] = df["GPA"].apply(parse_req_gpa)
    if "Difficulty" not in df.columns:
        df["Difficulty"] = 1.0

    for col in PROGRAM_FIELDS:
        if col not in df.columns:
            df[col] = "—"
    return df

table = load_data(CSV_FILE)
# ────────────────────────────────────────────────────────────


# ──────────────── GRADE UTILITIES ─────────────────

def tokenise(grades: str) -> List[str]:
    s = grades.upper().replace(" ", "")
    out, i = [], 0
    while i < len(s):
        if s[i:i+2] == "A*":
            out.append("A*"); i += 2
        else:
            out.append(s[i]); i += 1
    return out


def top_n_points(gs: List[str], n: int) -> int:
    pts = sorted([GRADE_POINTS.get(g, 0) for g in gs], reverse=True)
    return sum(pts[:n])


def percent_match(student: str, band: str, diff: float) -> float:
    if not band or str(band).strip() in ("-", "N/A"):
        return 0.0
    stu_pts = top_n_points(tokenise(student), len(tokenise(band)))
    req_pts = sum(GRADE_POINTS.get(g, 0) for g in tokenise(band)) * diff
    return round(100 * stu_pts / req_pts, 1) if req_pts else 0.0


def category_from_pct(p: float) -> str:
    if p >= 110: return "Safety"
    if p >= 95:  return "Match"
    return "Reach"
# ───────────────────────────────────────────────────


# ───────── GPT – ENRICHED ONE‑LINER COMMENTS ─────────

def gpt_batch_comment(rows: List[Dict]) -> Dict[int, str]:
    if not rows:
        return {}

    bullets = []
    for r in rows:
        enrich_parts = []
        if isinstance(r.get("Rank"), (int, float)) and not pd.isna(r["Rank"]):
            enrich_parts.append(f"Rank {r['Rank']}")
        if r.get("Co_op", "—") not in ("—", "N/A", ""):
            enrich_parts.append("co‑op yes" if str(r["Co_op"]).lower().startswith("y") else "co‑op no")
        if isinstance(r.get("Intake"), (int, float)) and not pd.isna(r["Intake"]):
            enrich_parts.append(f"intake {int(r['Intake'])}")
        if r.get("Tuition", "—") not in ("—", "N/A", ""):
            enrich_parts.append(f"tuition {r['Tuition']}")
        if isinstance(r.get("Intl_Pct"), (int, float)) and not pd.isna(r["Intl_Pct"]):
            enrich_parts.append(f"intl {int(r['Intl_Pct'])}%")
        enrich = ", ".join(enrich_parts) if enrich_parts else "—"
        bullets.append(
            f"{r['idx']} | {r['University']} | {r['Programme']} | Grades {r['grades']} vs {r['Band']} | "
            f"GPA {r['stu_gpa']:.2f} vs {r['req_gpa'] if not pd.isna(r['req_gpa']) else '—'} | {enrich}"
        )

    prompt = (
        "For EACH line, write ONE factual sentence (≤25 words) comparing the student's grades & GPA to the programme requirement and adding ONE programme fact if provided. "
        "Return exactly: id | comment\n\n" + "\n".join(bullets)
    )

    rsp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return id | comment only. No extra lines."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=MAX_COMP,
        reasoning_effort="low"
    )

    out: Dict[int, str] = {}
    for ln in rsp.choices[0].message.content.strip().splitlines():
        m = re.match(r"(\d+)\s*\|\s*(.+)", ln)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out
# ───────────────────────────────────────────────────────


# ────────────── UI HELPERS ────────────────────

def colour(cat: str) -> str:
    return {"Safety": "#d4edda", "Match": "#fff3cd", "Reach": "#f8d7da"}.get(cat, "#f8f9fa")


def badge(cat: str):
    icon = {"Safety": "✅", "Match": "🏅", "Reach": "🚀"}[cat]
    col  = {"Safety": "green", "Match": "orange", "Reach": "red"}[cat]
    st.markdown(
        f"<span style='background:{colour(cat)};color:{col};padding:0.2em 0.5em;border-radius:4px;font-weight:600;'>" \
        f"{icon} {cat}</span>", unsafe_allow_html=True)


def kpi_tiles(df: pd.DataFrame):
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Safety", df[df.Category == "Safety"].shape[0])
    col2.metric("🏅 Match",  df[df.Category == "Match"].shape[0])
    col3.metric("🚀 Reach",  df[df.Category == "Reach"].shape[0])

# ──────────────── WIZARD STATE ────────────────
if "step" not in st.session_state:
    st.session_state.step = 0
if "grades" not in st.session_state:
    st.session_state.grades = "A*A B"
if "gpa" not in st.session_state:
    st.session_state.gpa = 3.7
if "major" not in st.session_state:
    st.session_state.major = None

# ───────────────── STREAMLIT PAGE CONFIG ────────────────
st.set_page_config(page_title="Uni Admission Screener", page_icon="🎓", layout="centered")
st.title("🎓 University Admission Screener")
st.markdown("<style>#MainMenu{visibility:hidden} footer{visibility:hidden}</style>", unsafe_allow_html=True)

# ───────────────────── WIZARD FLOW ─────────────────────

def step_grades():
    st.header("Step 1 · Enter Your Academic Record")
    with st.form("grades_form"):
        grade_val = st.text_input("Your A-level grades (e.g. A*A B)", value=st.session_state.grades)
        gpa_val   = st.number_input("Your GPA (0-4 scale)", min_value=0.0, max_value=4.0,
                                   value=float(st.session_state.gpa), step=0.01, format="%.2f")
        submitted = st.form_submit_button("Next ➡️")
        if submitted:
            st.session_state.grades = grade_val
            st.session_state.gpa    = round(float(gpa_val), 2)
            st.session_state.step   = 1

def step_major():
    st.header("Step 2 · Select Programme / Major")
    majors = sorted(table.prog_norm.unique())
    st.session_state.major = st.selectbox("Programme", majors, index=majors.index(st.session_state.major) if st.session_state.major in majors else 0)
    col_prev, col_next = st.columns(2)
    col_prev.button("⬅️ Back", on_click=lambda: st.session_state.update(step=0))
    col_next.button("Next ➡️", on_click=lambda: st.session_state.update(step=2))

def step_results():
    st.header("Step 3 · Review Matches")
    subset = table[table.prog_norm == st.session_state.major]
    if subset.empty:
        st.warning("No programmes found for this major.")
        st.button("⬅️ Back", on_click=lambda: st.session_state.update(step=1))
        return

    rows: List[Dict] = []
    for i, row in subset.iterrows():
        band_raw = json.loads(row["Requirements (A-level)"]).get("overall_band", "")
        band = band_raw.strip()
        req_gpa = row["Req_GPA"]
        if re.search(r"[A-E]", band):
            pct = percent_match(st.session_state.grades, band, row["Difficulty"])
            cat = category_from_pct(pct)
        elif not pd.isna(req_gpa) and req_gpa > 0:
            pct = round(st.session_state.gpa / req_gpa * 100, 1)
            cat = category_from_pct(pct)
        else:
            pct, cat = 0.0, "N/A"
        rows.append({
            "idx": i,
            "grades": st.session_state.grades,
            "stu_gpa": st.session_state.gpa,
            "University": row["University"],
            "Programme": row["Major/Programme"],
            "Band": band if re.search(r"[A-E]", band) else "—",
            "req_gpa": req_gpa,
            "pct": pct,
            "Category": cat,
            "Rank": row["Rank"],
            "Co_op": row["Co_op"],
            "Intake": row["Intake"],
            "Tuition": row["Tuition"],
            "Intl_Pct": row["Intl_Pct"]
        })

    order = {"Safety": 0, "Match": 1, "Reach": 2, "N/A": 3}
    rows.sort(key=lambda r: (order.get(r["Category"], 99), -r["pct"]))
    comment_map = gpt_batch_comment(rows[:MAX_ROWS_FOR_GPT])
    df_res = pd.DataFrame(rows)

    kpi_tiles(df_res)
    st.divider()

    for _, r in df_res.iterrows():
        with st.container():
            st.markdown(f"<div style='background:{colour(r.Category)}; padding:1em; border-radius:8px;'>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='margin-bottom:0.3em'>{r.University} – {r.Programme}</h4>", unsafe_allow_html

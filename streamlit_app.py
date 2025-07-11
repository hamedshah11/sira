"""
Streamlit University Admission Screener — Wizard Edition
-------------------------------------------------------
* Features *
1. Multi‑step wizard (Grades → Major → Results → Download)
2. Enriched GPT comments: factual grade/GPA comparison **plus** one‑liner program facts
3. Card‑style results with category colour, icons & KPI summary
4. Resilient to missing columns (Rank, Co_op, Intake, Tuition, Intl_Pct)

Tested with streamlit ≥ 1.32, openai ≥ 1.25, model o3‑mini (adjust via env var OPENAI_MODEL)
"""

from __future__ import annotations
import os, json, re, pandas as pd, streamlit as st
from typing import List, Dict, Optional
from openai import OpenAI

# ─────────────────────────── CONFIG ────────────────────────────
CSV_FILE   = "university_requirements.csv"            # input dataset
MODEL_NAME = os.getenv("OPENAI_MODEL", "o3-mini")     # set in Secrets
MAX_COMP   = 1200                                     # GPT response room
MAX_ROWS_FOR_GPT = 25                                 # per‑search GPT rows
GRADE_POINTS = {"A*":56,"A":48,"B":40,"C":32,"D":24,"E":16}
PROGRAM_FIELDS = [  # optional enrich fields
    "Rank", "Co_op", "Intake", "Tuition", "Intl_Pct"
]
client = OpenAI()
# ───────────────────────────────────────────────────────────────


# ─────────────── DATA LOAD & NORMALISATION ───────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # canonical programme name (for filtering)
    df["prog_norm"] = (
        df["Major/Programme"].str.strip().str.lower()
          .str.replace(r"\s*\(.*\)", "", regex=True)
    )

    # add GPA requirement column (numeric)
    if "GPA" not in df.columns:
        df["GPA"] = "N/A"

    def parse_req_gpa(cell: str) -> Optional[float]:
        try:
            val = json.loads(cell).get("minimum", None)
            return float(val) if val not in (None, "", "N/A") else float("nan")
        except Exception:
            return float("nan")

    df["Req_GPA"] = df["GPA"].apply(parse_req_gpa)

    # ensure Difficulty column
    if "Difficulty" not in df.columns:
        df["Difficulty"] = 1.0

    # fill missing enrichment columns
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
    """Return {idx: comment} for each supplied row (≤ MAX_ROWS_FOR_GPT)."""
    if not rows:
        return {}

    # Compose bullet lines for GPT prompt
    bullets = []
    for r in rows:
        enrich_parts = []
        if isinstance(r.get("Rank"), (int, float)) and not pd.isna(r["Rank"]):
            enrich_parts.append(f"Rank {r['Rank']}")
        if r.get("Co_op", "—") not in ("—", "N/A", ""):
            enrich_parts.append("co-op yes" if str(r["Co_op"]).lower().startswith("y") else "co-op no")
        if isinstance(r.get("Intake"), (int, float)) and not pd.isna(r["Intake"]):
            enrich_parts.append(f"intake {int(r['Intake'])}")
        if r.get("Tuition", "—") not in ("—", "N/A", ""):
            enrich_parts.append(f"tuition {r['Tuition']}")
        if isinstance(r.get("Intl_Pct"), (int, float)) and not pd.isna(r["Intl_Pct"]):
            enrich_parts.append(f"intl {int(r['Intl_Pct'])}%")

        enrich_txt = ", ".join(enrich_parts) if enrich_parts else "—"
        bullets.append(
            f"{r['idx']} | {r['University']} | {r['Programme']} | "
            f"Grades {r['grades']} vs {r['Band']} | GPA {r['stu_gpa']:.2f} vs "
            f"{r['req_gpa'] if not pd.isna(r['req_gpa']) else '—'} | {enrich_txt}"
        )

    # Build prompt
    prompt = (
        "For EACH line below, write ONE factual sentence (≤25 words) "
        "comparing student's grades & GPA with programme requirements and mentioning ONE key programme fact if provided. "
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
    return {
        "Safety": "#d4edda",
        "Match": "#fff3cd",
        "Reach": "#f8d7da"
    }.get(cat, "#f8f9fa")


def badge(cat: str):
    icon = {
        "Safety": "✅",
        "Match": "🏅",
        "Reach": "🚀"
    }[cat]
    col = {
        "Safety": "green",
        "Match": "orange",
        "Reach": "red"
    }[cat]
    st.markdown(f"<span style='background:{colour(cat)};color:{col};padding:0.2em 0.5em;"
                f"border-radius:4px;font-weight:600;'>{icon} {cat}</span>", unsafe_allow_html=True)


def kpi_tiles(df: pd.DataFrame):
    safety = df[df.Category == "Safety"].shape[0]
    match  = df[df.Category == "Match"].shape[0]
    reach  = df[df.Category == "Reach"].shape[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Safety", safety)
    col2.metric("🏅 Match", match)
    col3.metric("🚀 Reach", reach)


# ──────────────── WIZARD STATE ────────────────
if "step" not in st.session_state:
    st.session_state.step = 0       # 0: grades, 1: major, 2: results, 3: download
if "grades" not in st.session_state:
    st.session_state.grades = "A*A B"
if "gpa" not in st.session_state:
    st.session_state.gpa = 3.7
if "major" not in st.session_state:
    st.session_state.major = None


def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step = max(0, st.session_state.step - 1)

# ───────────────── STREAMLIT PAGE CONFIG ────────────────
st.set_page_config(page_title="Uni Admission Screener", page_icon="🎓", layout="centered")
st.title("🎓 University Admission Screener")

# Hide default footer/menu for cleaner look
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ───────────────────── WIZARD FLOW ─────────────────────

if st.session_state.step == 0:
    st.header("Step 1 · Enter Your Academic Record")
    with st.form("grades_form"):
        st.session_state.grades = st.text_input("Your A‑level grades (e.g. A*A B)", st.session_state.grades)
        st.session_state.gpa = st.number_input("Your GPA (0‑4 scale)", 0.0, 4.0, st.session_state.gpa, 0.01, format="%.2f")
        submitted = st.form_submit_button("Next ➡️", on_click=next_step)
elif st.session_state.step == 1:
    st.header("Step 2 · Select Programme / Major")
    majors = sorted(table.prog_norm.unique())
    st.session_state.major = st.selectbox("Programme", majors, index=(majors.index(st.session_state.major) if st.session_state.major in majors else 0))
    col_prev, col_next = st.columns(2)
    with col_prev:
        st.button("⬅️ Back", on_click=prev_step)
    with col_next:
        st.button("Next ➡️", on_click=next_step)
elif st.session_state.step == 2:
    st.header("Step 3 · Review Matches")
    subset = table[table.prog_norm == st.session_state.major]
    if subset.empty:
        st.warning("No programmes found for this major.")
    else:
        # Build rows for comparison
        rows: List[Dict] = []
        for i, row in subset.iterrows():
            band_raw = json.loads(row["Requirements (A-level)"]).get("overall_band", "")
            band = band_raw.strip()
            req_gpa = row["Req_GPA"]

            # Determine match percentage and category
            if re.search(r"[A-E]", band):
                pct = percent_match(st.session_state.grades, band, row["Difficulty"])
                cat = category_from_pct(pct)
            elif not pd.isna(req_gpa) and req_gpa > 0:
                pct = round(st.session_state.gpa / req_gpa * 100, 1)
                cat = category_from_pct(pct)
            else:
                pct, cat = 0.0, "N/A"

            # Assemble row dict
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

        # GPT comments for top rows
        comment_map = gpt_batch_comment(rows[:MAX_ROWS_FOR_GPT])
        df_res = pd.DataFrame(rows)

        # KPIs
        kpi_tiles(df_res)
        st.divider()

        # Result cards
        for _, r in df_res.iterrows():
            with st.container():
                st.markdown(f"<div style='background:{colour(r.Category)}; padding:1em; border-radius:8px;'>", unsafe_allow_html=True)

                # Header with university & programme
                st.markdown(f"<h4 style='margin-bottom:0.3em'>{r.University} – {r.Programme}</h4>", unsafe_allow_html=True)
                badge(r.Category)

                # GPA / Grade comparison bar
                bar_val = max(0.0, min(r.pct/100.0, 1.0))  # clamp 0–1
                st.progress(bar_val, text=f"{r.pct}% of requirement met")

                # Programme facts line
                facts = []
                if pd.notna(r.Rank) and str(r.Rank) not in ("—", "nan"):
                    facts.append(f"🏆 Rank: {r.Rank}")
                if str(r.Co_op).lower().startswith("y"):
                    facts.append("💼 Co-op: Yes")
                if pd.notna(r.Intake) and str(r.Intake) not in ("—", "nan"):
                    facts.append(f"👥 Intake: {int(r.Intake)}")
                if str(r.Tuition) not in ("—", "", "nan"):
                    facts.append(f"💰 Tuition: {r.Tuition}")
                if pd.notna(r.Intl_Pct) and str(r.Intl_Pct) not in ("—", "nan"):
                    facts.append(f"🌍 Intl: {int(r.Intl_Pct)}%")
                if facts:
                    st.markdown(" | ".join(facts))

                # GPT comment
                st.caption(comment_map.get(r.idx, "— no comment —"))
                st.markdown("</div>", unsafe_allow_html=True)
                st.write("")

        # Navigation / download
        st.divider()
        col_prev, col_next = st.columns([1, 2])
        with col_prev:
            st.button("⬅️ Back", on_click=prev_step)
        with col_next:
            st.session_state.df_res = df_res  # store for download
            st.button("Next ➡️", on_click=next_step, disabled=df_res.empty)
elif st.session_state.step == 3:
    st.header("Step 4 · Download Your Results")
    if "df_res" not in st.session_state or st.session_state.df_res.empty:
        st.error("No results to download. Go back and run a search first.")
    else:
        st.success("Your personalized results are ready! ✨")
        csv_bytes = st.session_state.df_res.drop(columns=["idx"]).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv_bytes, file_name="uni_matches.csv", mime="text/csv")
    st.button("⬅️ Start Over", on_click=lambda: st.session_state.update(dict(step=0)))

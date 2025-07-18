# streamlit_app.py ─ holistic screener with GPA & per-row GPT comments
# Tested with: streamlit ≥ 1.32 • openai ≥ 1.25 • o3-mini model.

import os, json, re
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# ───────────────────────── CONFIG ─────────────────────────
CSV_FILE   = "university_requirements.csv"           # dataset
MODEL_NAME = os.getenv("OPENAI_MODEL", "o3-mini")    # override in Secrets if needed
MAX_COMP   = 1_200                                   # GPT token budget
MAX_ROWS_FOR_GPT = 25                                # rows sent for comments
GRADE_POINTS = {"A*": 56, "A": 48, "B": 40, "C": 32, "D": 24, "E": 16}

# Initialise OpenAI client (expects OPENAI_API_KEY in secrets or env var)
client = OpenAI()

# ──────────────────────────────────────────────────────────
# ▸ DATA LOAD & NORMALISE
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalised programme name (lower-case, no brackets)
    df["prog_norm"] = (
        df["Major/Programme"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s*\(.*\)", "", regex=True)
    )

    # GPA column may be missing in the raw file
    if "GPA" not in df.columns:
        df["GPA"] = "N/A"

    # Parse required GPA (stored as JSON in the GPA column)
    def parse_req_gpa(cell) -> Optional[float]:
        try:
            obj = json.loads(cell)
            val = obj.get("minimum")
            return float(val) if val not in (None, "", "N/A") else float("nan")
        except Exception:
            return float("nan")

    df["Req_GPA"] = df["GPA"].apply(parse_req_gpa)

    # Difficulty (multiplier) defaults to 1.0 if absent
    if "Difficulty" not in df.columns:
        df["Difficulty"] = 1.0

    return df


table = load_data(CSV_FILE)

# ──────────────────────────────────────────────────────────
# ▸ GRADE HELPERS & CATEGORIES
def tokenise(txt: str) -> List[str]:
    """Split a grade string like 'A*A B' ⇒ ['A*', 'A', 'B']"""
    s = txt.upper().replace(" ", "")
    out, i = [], 0
    while i < len(s):
        if s[i : i + 2] == "A*":
            out.append("A*")
            i += 2
        else:
            out.append(s[i])
            i += 1
    return out


def top_n_points(gs: List[str], n: int) -> int:
    pts = sorted((GRADE_POINTS.get(g, 0) for g in gs), reverse=True)
    return sum(pts[:n])


def percent_match(student: str, band: str, diff: float) -> float:
    """Return %-match of a student's top grades vs the requirement band."""
    if not band or str(band).strip() in ("-", "N/A"):
        return 0.0
    stu_pts = top_n_points(tokenise(student), len(tokenise(band)))
    req_pts = sum(GRADE_POINTS.get(g, 0) for g in tokenise(band)) * diff
    return round(100.0 * stu_pts / req_pts, 1) if req_pts else 0.0


def category_from_pct(p: float) -> str:
    if p >= 110:
        return "Safety"
    if p >= 95:
        return "Match"
    return "Reach"


# ──────────────────────────────────────────────────────────
# ▸ GPT-BASED ONE-LINE COMMENTS
def gpt_batch_comment(rows: List[Dict]) -> Dict[int, str]:
    """Ask GPT for a <20-word factual comparison line for each row id."""
    if not rows:
        return {}

    bullets = [
        (
            f"{r['idx']} | {r['University']} | {r['Programme']} | "
            f"Student {r['grades']} GPA {r['stu_gpa']:.2f} | "
            f"Req {r['Band']} "
            f"GPA {r['req_gpa'] if not pd.isna(r['req_gpa']) else '—'}"
        )
        for r in rows
    ]

    prompt = (
        "For EACH line, write ONE factual comparison (<20 words) of the student's "
        "grades & GPA vs the programme requirements. No advice. "
        "Return exactly: id | comment.\n\n"
        + "\n".join(bullets)
    )

    rsp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return id | comment only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_COMP,          # renamed in v1 API
        temperature=0.0,              # deterministic, factual
    )

    out: Dict[int, str] = {}
    for ln in rsp.choices[0].message.content.strip().splitlines():
        m = re.match(r"(\d+)\s*\|\s*(.+)", ln)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out


# ──────────────────────────────────────────────────────────
# ▸ STREAMLIT VISUAL HELPERS
def colour(c: str) -> str:
    return {
        "Safety": "#d4edda",
        "Match": "#fff3cd",
        "Reach": "#f8d7da",
    }.get(c, "#f8f9fa")


def badge(c: str):
    icon = {
        "Safety": ":material/check_circle:",
        "Match": ":material/balance:",
        "Reach": ":material/rocket_launch:",
    }[c]
    colr = {"Safety": "green", "Match": "orange", "Reach": "red"}[c]
    return st.badge(c, icon=icon, color=colr)


def kpis(df: pd.DataFrame):
    a, b, c = st.columns(3)
    a.metric("Safety ✅", df[df.Category == "Safety"].shape[0])
    b.metric("Match 🎯", df[df.Category == "Match"].shape[0])
    c.metric("Reach 🚀", df[df.Category == "Reach"].shape[0])


# ──────────────────────────────────────────────────────────
# ▸ STREAMLIT APP
st.set_page_config("University Screener", "🎓")
st.title("🎓 University Admission Screener")

stu_gr = st.text_input("Your A-level grades (e.g. A*A B)", "A*A B")
stu_gpa = st.number_input("Your GPA (0-4 scale)", 0.0, 4.0, 3.7, 0.01)
major = st.selectbox("Programme / Major", sorted(table.prog_norm.unique()))

if st.button("🔍 Search") and stu_gr.strip():
    subset = table[table.prog_norm == major]

    if subset.empty:
        st.warning("No programmes found for that major.")
        st.stop()

    # Build result rows
    rows: List[Dict] = []
    for i, row in subset.iterrows():
        # Parse A-level band safely
        try:
            band_raw = json.loads(row["Requirements (A-level)"]).get("overall_band", "")
            band = str(band_raw).strip()
        except Exception:
            band = ""

        req_gpa = row["Req_GPA"]

        # 1) Match on A-level band
        if re.search(r"[A-E]", band):
            pct = percent_match(stu_gr, band, row["Difficulty"])
            cat = category_from_pct(pct)

        # 2) Else, fallback to GPA
        elif not pd.isna(req_gpa) and req_gpa > 0:
            pct = round((stu_gpa / req_gpa) * 100, 1)
            cat = category_from_pct(pct)

        # 3) No comparable requirement
        else:
            pct = 0.0
            cat = "N/A"

        rows.append(
            dict(
                idx=i,
                grades=stu_gr,
                stu_gpa=stu_gpa,
                University=row["University"],
                Programme=row["Major/Programme"],
                Band=band if re.search(r"[A-E]", band) else "—",
                req_gpa=req_gpa,
                pct=pct,
                Category=cat,
            )
        )

    # Sort by category then descending match %
    order = {"Safety": 0, "Match": 1, "Reach": 2, "N/A": 3}
    rows.sort(key=lambda r: (order.get(r["Category"], 99), -r["pct"]))

    # Ask GPT for <= 25 comparison comments
    comment_map = gpt_batch_comment(rows[:MAX_ROWS_FOR_GPT])

    # Convert to DataFrame for display / download
    df_res = pd.DataFrame(rows)

    # KPI cards
    kpis(df_res)

    # Category tabs
    tab_titles = ["✅ Safety", "🎯 Match", "🚀 Reach", "ℹ️ N/A"]
    tabs = st.tabs(tab_titles)
    tab_map = dict(zip(["Safety", "Match", "Reach", "N/A"], tabs))

    for cat in ["Safety", "Match", "Reach", "N/A"]:
        with tab_map[cat]:
            cat_rows = df_res[df_res.Category == cat]
            if cat_rows.empty:
                st.info("No programmes in this category.")
                continue

            for _, r in cat_rows.iterrows():
                with st.container():
                    st.markdown(
                        f'<div style="background-color:{colour(cat)};padding:8px;border-radius:6px">',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{r.University} – {r.Programme}**")
                    badge(cat)

                    bar = max(0.0, min(r.pct / 100.0, 1.0))
                    st.progress(
                        bar,
                        text=f"{r.pct}% match • GPA req "
                        f"{'—' if pd.isna(r.req_gpa) else r.req_gpa}",
                    )

                    with st.expander("💬 GPT comparison"):
                        st.write(comment_map.get(r.idx, "— no comment —"))

                    st.markdown("</div>", unsafe_allow_html=True)
                    st.write("")

    st.download_button(
        "📥 Download CSV",
        df_res.drop(columns=["idx"]).to_csv(index=False),
        file_name="uni_matches.csv",
        mime="text/csv",
    )
else:
    st.info("Enter your grades & GPA, choose a major, then click **Search**.")

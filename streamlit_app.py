# streamlit_app.py — holistic screener with GPA & per-row GPT comments
# Compatible with streamlit ≥1.32 • openai ≥1.25 • model: o3-mini
# Enhanced with IB grade support and GPA conversions

import os, json, re
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI                                   # pip install --upgrade openai

# ─── CONFIG ──────────────────────────────────────────────────────────
CSV_FILE         = "university_requirements.csv"            # dataset path
MODEL_NAME       = os.getenv("OPENAI_MODEL", "o3-mini")
MAX_COMP         = 1_200                                    # completion-token budget
MAX_ROWS_FOR_GPT = 25                                       # rows per prompt
GRADE_POINTS     = {"A*":56,"A":48,"B":40,"C":32,"D":24,"E":16}

# IB to GPA conversion (standard 4.0 scale)
IB_TO_GPA = {
    45: 4.0, 44: 4.0, 43: 4.0, 42: 3.9, 41: 3.8, 40: 3.7,
    39: 3.6, 38: 3.5, 37: 3.4, 36: 3.3, 35: 3.2, 34: 3.1,
    33: 3.0, 32: 2.9, 31: 2.8, 30: 2.7, 29: 2.6, 28: 2.5,
    27: 2.4, 26: 2.3, 25: 2.2, 24: 2.1
}

# Grade conversion mappings
ALEVEL_GPA_VALUES = {"A*": 4.0, "A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "E": 0.0}
IB_GRADE_TO_GPA = {7: 4.0, 6: 3.7, 5: 3.3, 4: 2.7, 3: 2.3, 2: 1.7, 1: 1.0}

client = OpenAI()  # requires OPENAI_API_KEY in env or Streamlit secrets

# ─── DATA LOAD & CLEAN ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["prog_norm"] = (
        df["Major/Programme"]
        .astype(str).str.strip().str.lower()
        .str.replace(r"\s*\(.*\)", "", regex=True)
    )

    if "GPA" not in df.columns:
        df["GPA"] = "N/A"

    def parse_req_gpa(cell) -> Optional[float]:
        try:
            val = json.loads(cell).get("minimum")
            return float(val) if val not in (None,"","N/A") else float("nan")
        except Exception:
            return float("nan")

    df["Req_GPA"] = df["GPA"].apply(parse_req_gpa)
    if "Difficulty" not in df.columns:
        df["Difficulty"] = 1.0
    return df

table = load_data(CSV_FILE)

# ─── GRADE CONVERSION FUNCTIONS ──────────────────────────────────────
def convert_ib_grades_to_gpa(ib_grades_str: str) -> float:
    """Convert IB subject grades (e.g., '7,6,5,6,7,4') to GPA (4.0 scale)"""
    try:
        # Parse grades from string (comma-separated or space-separated)
        grades_str = re.sub(r'[^\d,\s]', '', ib_grades_str)  # Keep only digits, commas, spaces
        grades = [int(g.strip()) for g in re.split(r'[,\s]+', grades_str) if g.strip().isdigit()]
        
        if not grades:
            return 2.0
        
        # Convert each grade to GPA and calculate average
        gpa_values = [IB_GRADE_TO_GPA.get(grade, 1.0) for grade in grades if 1 <= grade <= 7]
        
        if not gpa_values:
            return 2.0
            
        return round(sum(gpa_values) / len(gpa_values), 2)
    except:
        return 2.0

def convert_alevel_to_gpa(grades: str) -> float:
    """Convert A-level grades to GPA using standard conversion"""
    grade_list = tokenise(grades)
    if not grade_list:
        return 2.0
    
    # Convert each grade to GPA value and calculate average
    gpa_values = [ALEVEL_GPA_VALUES.get(grade, 0.0) for grade in grade_list]
    
    if not gpa_values:
        return 2.0
        
    return round(sum(gpa_values) / len(gpa_values), 2)

def get_student_gpa(grade_type: str, grades_input: str, gpa_input: float) -> Tuple[float, str]:
    """Get student's GPA and display string based on input type"""
    if grade_type == "IB":
        converted_gpa = convert_ib_grades_to_gpa(grades_input)
        return converted_gpa, f"IB grades {grades_input} (≈{converted_gpa:.2f} GPA)"
    elif grade_type == "A-level":
        converted_gpa = convert_alevel_to_gpa(grades_input)
        return converted_gpa, f"A-level {grades_input} (≈{converted_gpa:.2f} GPA)"
    else:  # Direct GPA
        return gpa_input, f"GPA {gpa_input:.2f}"

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────
def tokenise(txt: str) -> List[str]:
    s, out, i = txt.upper().replace(" ", ""), [], 0
    while i < len(s):
        out.append("A*" if s[i:i+2]=="A*" else s[i]); i += 2 if s[i:i+2]=="A*" else 1
    return out

def top_n_points(gs: List[str], n:int)->int:
    return sum(sorted((GRADE_POINTS.get(g,0) for g in gs), reverse=True)[:n])

def percent_match(student:str, band:str, diff:float)->float:
    if not band or band.strip() in ("-","N/A"): return 0.0
    stu_pts = top_n_points(tokenise(student), len(tokenise(band)))
    req_pts = sum(GRADE_POINTS.get(g,0) for g in tokenise(band))*diff
    return round(100*stu_pts/req_pts,1) if req_pts else 0.0

def gpa_match_percent(student_gpa: float, req_gpa: float) -> float:
    """Calculate percentage match based on GPA"""
    if pd.isna(req_gpa) or req_gpa <= 0:
        return 0.0
    return round(100 * student_gpa / req_gpa, 1)

def category_from_pct(p:float)->str:
    return "Safety" if p>=110 else "Match" if p>=95 else "Reach"

# ─── GPT BATCH COMMENT ───────────────────────────────────────────────
def gpt_batch_comment(rows:List[Dict])->Dict[int,str]:
    if not rows: return {}

    bullets = [
        f"{r['idx']} | {r['University']} | {r['Programme']} | "
        f"Student {r['student_display']} | "
        f"Req A-level {r['Band']} GPA {r['req_gpa'] if not pd.isna(r['req_gpa']) else '—'}"
        for r in rows
    ]
    prompt = ("For EACH line, write ONE factual comparison (<20 words) of the student's "
              "performance vs the programme requirements. No advice. "
              "Return exactly: id | comment.\n\n" + "\n".join(bullets))

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":"Return id | comment only."},
                  {"role":"user","content":prompt}],
        max_completion_tokens=MAX_COMP,
        reasoning_effort="low"
    )

    out: Dict[int,str] = {}
    for ln in resp.choices[0].message.content.strip().splitlines():
        m = re.match(r"(\d+)\s*\|\s*(.+)", ln)
        if m: out[int(m.group(1))] = m.group(2).strip()
    return out

# ─── UI HELPERS ──────────────────────────────────────────────────────
def colour(c): return {"Safety":"#d4edda","Match":"#fff3cd","Reach":"#f8d7da"}.get(c,"#f8f9fa")
def badge(c):
    icon = {"Safety":":material/check_circle:","Match":":material/balance:","Reach":":material/rocket_launch:"}[c]
    colr = {"Safety":"green","Match":"orange","Reach":"red"}[c]
    return st.badge(c, icon=icon, color=colr)
def kpis(df):
    a,b,c = st.columns(3)
    a.metric("Safety ✅", df[df.Category=="Safety"].shape[0])
    b.metric("Match 🎯",  df[df.Category=="Match"].shape[0])
    c.metric("Reach 🚀",  df[df.Category=="Reach"].shape[0])

# ─── STREAMLIT APP ───────────────────────────────────────────────────
st.set_page_config("Uni Screener","🎓")
st.title("🎓 University Admission Screener")

# Grade input type selection
grade_type = st.radio(
    "Select your qualification type:",
    ["A-level", "IB", "Direct GPA"],
    horizontal=True
)

# Input fields based on selected type
if grade_type == "A-level":
    stu_gr = st.text_input("Your A-level grades (e.g. A*A B)", "A*A B")
    stu_gpa = st.number_input("Your GPA (0-4 scale, optional - will convert from A-levels)", 0.0, 4.0, 0.0, 0.01)
elif grade_type == "IB":
    stu_gr = st.text_input("Your IB subject grades (e.g. 7,6,5,6,7,4 or 7 6 5 6 7 4)", "7,6,5,6,7,4")
    stu_gpa = st.number_input("Your GPA (0-4 scale, optional - will convert from IB)", 0.0, 4.0, 0.0, 0.01)
else:  # Direct GPA
    stu_gr = ""
    stu_gpa = st.number_input("Your GPA (0-4 scale)", 0.0, 4.0, 3.7, 0.01)

major = st.selectbox("Programme / Major", sorted(table.prog_norm.unique()))

if st.button("🔍 Search") and (stu_gr.strip() or grade_type == "Direct GPA"):
    subset = table[table.prog_norm == major]
    if subset.empty:
        st.warning("No programmes found."); st.stop()

    # Get student's effective GPA and display string
    student_gpa, student_display = get_student_gpa(grade_type, stu_gr, stu_gpa)

    rows=[]
    for i,row in subset.iterrows():
        try:
            band = json.loads(row["Requirements (A-level)"]).get("overall_band","").strip()
        except Exception: 
            band=""
        req_gpa = row["Req_GPA"]

        # Determine comparison method and calculate match percentage
        if grade_type == "IB":
            # For IB: Only compare GPA
            if not pd.isna(req_gpa) and req_gpa > 0:
                pct = gpa_match_percent(student_gpa, req_gpa)
                cat = category_from_pct(pct)
            else:
                pct, cat = 0.0, "N/A"
        else:
            # For A-level: Compare both A-level grades and GPA, take the better match
            alevel_pct = 0.0
            gpa_pct = 0.0
            
            if re.search(r"[A-E]", band) and grade_type == "A-level":
                alevel_pct = percent_match(stu_gr, band, row["Difficulty"])
            
            if not pd.isna(req_gpa) and req_gpa > 0:
                gpa_pct = gpa_match_percent(student_gpa, req_gpa)
            
            # Take the higher percentage for better matching
            pct = max(alevel_pct, gpa_pct)
            cat = category_from_pct(pct) if pct > 0 else "N/A"

        rows.append(dict(
            idx=i, student_display=student_display,
            University=row["University"], Programme=row["Major/Programme"],
            Band=band if re.search(r"[A-E]", band) else "—",
            req_gpa=req_gpa, pct=pct, Category=cat
        ))

    rows.sort(key=lambda r:({"Safety":0,"Match":1,"Reach":2,"N/A":3}[r["Category"]], -r["pct"]))
    comment_map = gpt_batch_comment(rows[:MAX_ROWS_FOR_GPT])
    df_res = pd.DataFrame(rows)

    st.success(f"Analyzed {len(rows)} programmes for {student_display}")
    kpis(df_res)

    tabs = st.tabs(["✅ Safety","🎯 Match","🚀 Reach","ℹ️ N/A"])
    for cat,tab in zip(["Safety","Match","Reach","N/A"], tabs):
        with tab:
            cr = df_res[df_res.Category==cat]
            if cr.empty: 
                st.info("No programmes in this category.")
                continue
            
            for _,r in cr.iterrows():
                with st.container():
                    st.markdown(f'<div style="background-color:{colour(cat)};padding:8px;border-radius:6px">',unsafe_allow_html=True)
                    st.markdown(f"**{r.University} – {r.Programme}**")
                    badge(cat)
                    
                    req_display = []
                    if r.Band != "—":
                        req_display.append(f"A-level {r.Band}")
                    if not pd.isna(r.req_gpa):
                        req_display.append(f"GPA {r.req_gpa}")
                    req_text = " • ".join(req_display) if req_display else "No specific requirements"
                    
                    st.progress(max(0,min(r.pct/100,1)), text=f"{r.pct}% Match • Req: {req_text}")
                    with st.expander("💬 GPT comparison"):
                        st.write(comment_map.get(r.idx,"— no comment —"))
                    st.markdown("</div>",unsafe_allow_html=True)
                    st.write("")

    st.download_button("📥 Download CSV", df_res.drop(columns=["idx"]).to_csv(index=False), "uni_matches.csv", "text/csv")
else:
    if grade_type == "Direct GPA":
        st.info("Enter your GPA, choose a major, then click Search.")
    else:
        st.info("Enter your grades, choose a major, then click Search.")

# Help section
with st.expander("ℹ️ How it works"):
    st.markdown("""
    **Grade Conversions:**
    - **A-level**: Converts to approximate GPA and compares against both A-level requirements and GPA requirements
    - **IB**: Converts total score to GPA and compares only against GPA requirements  
    - **Direct GPA**: Uses your GPA as-is for comparison
    
    **Matching Categories:**
    - **Safety** (≥110%): You exceed requirements significantly
    - **Match** (95-109%): Your performance closely matches requirements  
    - **Reach** (<95%): Requirements are above your current performance
    
    **Note**: Conversions are approximate. Always check official university requirements.
    """)

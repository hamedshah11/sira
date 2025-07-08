import os, json
import pandas as pd
import streamlit as st
from typing import List

# ---------------- CONFIG ----------------
DATA_FILE    = "university_requirements.csv"
GRADE_MAP    = {"A*":6,"A":5,"B":4,"C":3,"D":2,"E":1}
OPENAI_MODEL = os.getenv("OPENAI_MODEL","o3-mini")
# ----------------------------------------

# ---------- DATA LOADER (cache-bust) ----
@st.cache_data
def _load_data(path: str, _mtime: float):
    df = pd.read_csv(path)
    df["Programme_norm"] = (df["Major/Programme"]
        .str.strip()
        .str.lower()
        .str.replace(r"\s*\(.*\)","",regex=True))
    return df

def get_data():
    mtime = os.path.getmtime(DATA_FILE)
    return _load_data(DATA_FILE, mtime)
# ----------------------------------------

# ---------- RULES ENGINE ---------------
def parse_grades(txt:str)->List[str]:
    txt = txt.upper().replace(" ","")
    i,out=0,[]
    while i<len(txt):
        if txt[i]=="A" and i+1<len(txt) and txt[i+1]=="*":
            out.append("A*"); i+=2
        else:
            out.append(txt[i]); i+=1
    return out

def numeric(gs): return sorted([GRADE_MAP.get(g,0) for g in gs],reverse=True)

def tag(stu,band):
    if not band: return "N/A"
    s,n = numeric(parse_grades(stu)),numeric(parse_grades(band))
    s+= [0]*(len(n)-len(s)); n+= [0]*(len(s)-len(n))
    for a,b in zip(s,n):
        if a>b: return "Safety"
        if a<b: return "Reach"
    return "Match"
# ---------------------------------------

# ---------- LLM HELPER with RAW JSON ----
def llm(prompt:str)->str:
    import openai, textwrap
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY","")
    if not key: raise RuntimeError("Add OPENAI_API_KEY to Secrets.")
    openai.api_key = key
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"user","content":prompt}],
        max_completion_tokens=350
    )

    # Dump raw JSON in sidebar
    st.sidebar.expander("🔍 Raw LLM response").write(response)

    if not response.choices or not response.choices[0].message:
        return ""
    return response.choices[0].message.content.strip()
# ----------------------------------------

def main():
    st.set_page_config("Uni Screener","🎓")
    st.title("🎓 University Admission Screener")

    df  = get_data()
    grd = st.text_input("Your A-level grades","A*A B")
    majors = sorted(df["Programme_norm"].unique())
    maj = st.selectbox("Programme",majors,index=0)

    if st.button("Search") and grd:
        sub = df[df["Programme_norm"]==maj]
        if sub.empty:
            st.warning("None found"); st.stop()
        r=[]
        for _,row in sub.iterrows():
            band = ""
            try: band = json.loads(row["Requirements (A-level)"]).get("overall_band","")
            except: pass
            r.append({"University":row["University"],
                      "Programme":row["Major/Programme"],
                      "Band":band,
                      "Category":tag(grd,band)})
        out = pd.DataFrame(r)
        st.dataframe(out,use_container_width=True)

        bullets="\n".join(f"[{c}] {u} – {p} (needs {b})"
                          for u,p,b,c in out[["University","Programme","Band","Category"]].itertuples(index=False))
        prompt=(f'Grades: "{grd}" | Major: "{maj}"\n{bullets}\n'
                "Explain tags and give tips.")
        with st.spinner("GPT…"):
            txt = llm(prompt)
        if txt:
            st.markdown("### GPT Advice")
            st.markdown(txt or "_Empty completion_")
        else:
            st.error("LLM returned empty response — check sidebar for raw JSON.")

if __name__=="__main__":
    main()

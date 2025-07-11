import streamlit as st
import pandas as pd
import re

# Page configuration
st.set_page_config(page_title="University Admission Screener", page_icon="🎓", layout="centered")
st.title("University Admission Screener")

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1

@st.cache_data
def load_data():
    # Load the dataset (adjust filename as needed)
    return pd.read_csv("university_requirements.csv")

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Normalize expected column names if needed
df_columns = list(df.columns)
# Rename normalized major column if present under a different name
for col in df_columns:
    if "ormalized" in col and "ajor" in col:
        df.rename(columns={col: "Normalized Major"}, inplace=True)
        break
# Rename A-level requirement column if named differently
for col in df_columns:
    if col.lower().startswith("a-level"):
        df.rename(columns={col: "A-level Requirements"}, inplace=True)
        break
# Rename GPA requirement column if named differently
for col in df_columns:
    if col.lower().startswith("gpa"):
        df.rename(columns={col: "GPA Requirements"}, inplace=True)
        break

def step_grades():
    st.header("Step 1: Academic Record")
    with st.form("grades_form"):
        grades_input = st.text_input("Your A-level grades (e.g. A*A B)")
        gpa_input = st.number_input("Your GPA (0–4 scale)", min_value=0.0, max_value=4.0, step=0.01)
        next_btn = st.form_submit_button("Next")
        if next_btn:
            if grades_input.strip() == "":
                st.error("Please enter your A-level grades.")
            else:
                st.session_state.grades = grades_input.strip()
                st.session_state.gpa = float(gpa_input)
                st.session_state.step = 2

def step_major():
    st.header("Step 2: Major Selection")
    if df is None or df.empty:
        st.error("Majors list is not available.")
        return
    majors = sorted(df["Normalized Major"].dropna().unique())
    default_index = 0
    if "major" in st.session_state:
        try:
            default_index = majors.index(st.session_state.major)
        except ValueError:
            default_index = 0
    selected_major = st.selectbox("Choose a programme/major:", majors, index=default_index)
    if st.button("Search"):
        st.session_state.major = selected_major
        st.session_state.step = 3

def gpt_batch_comment(rows):
    comments = []
    try:
        import openai
        if hasattr(st, "secrets") and "openai_key" in st.secrets:
            openai.api_key = st.secrets["openai_key"]
        else:
            raise RuntimeError("OpenAI API key not found.")
        system_msg = {"role": "system", "content": "You are an assistant that evaluates a student's fit for a university programme based on their grades and the programme requirements."}
        for _, row in rows.iterrows():
            uni = row.get("University", "the university")
            prog = row.get("Programme", row.get("Program", "the programme"))
            req_a = row.get("A-level Requirements", None)
            req_g = row.get("GPA Requirements", None)
            student_grades = st.session_state.grades
            student_gpa = st.session_state.gpa
            if (pd.isna(req_a) or req_a in [None, "", "N/A"]) and (pd.isna(req_g) or req_g in [None, "", "N/A"]):
                prompt = f"The student has A-level grades {student_grades} and GPA {student_gpa}. They are applying to {prog} at {uni}, but there are no specific admission requirements listed for this programme."
            elif pd.isna(req_a) or req_a in [None, "", "N/A"]:
                prompt = f"The student has A-level grades {student_grades} and GPA {student_gpa}, applying to {prog} at {uni}. The programme has a GPA requirement of {float(req_g):.2f}."
            elif pd.isna(req_g) or req_g in [None, "", "N/A"]:
                prompt = f"The student has A-level grades {student_grades} and GPA {student_gpa}, applying to {prog} at {uni}. The programme requires {req_a} at A-level."
            else:
                try:
                    req_g_val = float(req_g)
                    prompt = f"The student has A-level grades {student_grades} and GPA {student_gpa}, applying to {prog} at {uni}. The programme requires {req_a} at A-level and a GPA of {req_g_val:.2f}."
                except:
                    prompt = f"The student has A-level grades {student_grades} and GPA {student_gpa}, applying to {prog} at {uni}. The programme requires {req_a} at A-level and a GPA of {req_g}."
            user_msg = {"role": "user", "content": prompt + " Provide a brief assessment of how well the student meets these requirements."}
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[system_msg, user_msg],
                max_tokens=100,
                temperature=0.7
            )
            reply = response["choices"][0]["message"]["content"].strip()
            comments.append(reply)
    except Exception as e:
        # On any error, output debugging info and fallback comments
        st.write(f"GPT comment generation error: {e}")
        for _ in range(len(rows)):
            comments.append("N/A")
    return comments

def step_results():
    st.header("Step 3: Programme Results")
    if "grades" not in st.session_state or "gpa" not in st.session_state or "major" not in st.session_state:
        st.error("Missing information from previous steps.")
        return
    major = st.session_state.major
    filtered = df[df["Normalized Major"] == major].copy()
    if filtered.empty:
        st.warning("No programmes found for the selected major.")
        if st.button("Back"):
            st.session_state.step = 2
        return
    grade_points = {"A*": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
    def parse_alevels(grades_str):
        if pd.isna(grades_str) or not isinstance(grades_str, str):
            return None
        tokens = re.findall(r"A\*|A|B|C|D|E", grades_str.upper())
        if not tokens:
            return None
        return sum(grade_points.get(t, 0) for t in tokens) or None

    user_grade_points = parse_alevels(st.session_state.grades)
    user_gpa = st.session_state.gpa
    categories = []
    percent_matches = []

    for idx, row in filtered.iterrows():
        req_a = row.get("A-level Requirements", None)
        req_g = row.get("GPA Requirements", None)
        req_points = parse_alevels(req_a)
        ratios = []
        if req_points is not None and user_grade_points is not None and req_points > 0:
            ratios.append(user_grade_points / req_points)
        if pd.notna(req_g) and req_g not in [None, 0, 0.0]:
            try:
                req_g_val = float(req_g)
            except:
                req_g_val = None
            if req_g_val and req_g_val > 0:
                ratios.append(user_gpa / req_g_val)
        percent_match = None
        if ratios:
            percent_match = (sum(ratios) / len(ratios)) * 100.0
            if percent_match > 150.0:
                percent_match = 150.0
        # Determine category
        if percent_match is None:
            category = "N/A"
        elif percent_match >= 120.0:
            category = "Safety"
        elif percent_match >= 100.0:
            category = "Match"
        else:
            category = "Reach"
        categories.append(category)
        percent_matches.append(percent_match)
        filtered.at[idx, "Category"] = category
        filtered.at[idx, "Percent Match"] = percent_match

    comments = gpt_batch_comment(filtered)
    filtered["Comment"] = comments

    safety_count = categories.count("Safety")
    match_count = categories.count("Match")
    reach_count = categories.count("Reach")
    na_count = categories.count("N/A")

    col1, col2, col3 = st.columns(3)
    col1.metric("Safety", safety_count)
    col2.metric("Match", match_count)
    col3.metric("Reach", reach_count)

    tabs_list = [
        f"✅ Safety ({safety_count})",
        f"🥈 Match ({match_count})",
        f"🚀 Reach ({reach_count})"
    ]
    if na_count > 0:
        tabs_list.append(f"📓 N/A ({na_count})")
    category_tabs = st.tabs(tabs_list)

    # Safety tab
    with category_tabs[0]:
        if safety_count == 0:
            st.write("No programs found in this category.")
        else:
            for _, row in filtered[filtered["Category"] == "Safety"].iterrows():
                uni = row.get("University", "")
                prog = row.get("Programme", row.get("Program", ""))
                st.markdown(f"**{uni} – {prog}**")
                st.write("✅ **Safety**")
                pct = row["Percent Match"]
                req_a = row.get("A-level Requirements", None)
                req_g = row.get("GPA Requirements", None)
                match_text = "N/A" if pct is None else f"{pct:.1f}% Match"
                req_parts = []
                if isinstance(req_a, str) and req_a.strip():
                    req_parts.append(f"A-level req {req_a}")
                if pd.notna(req_g):
                    try:
                        req_val = float(req_g)
                        req_parts.append(f"GPA req {req_val:.2f}")
                    except:
                        req_parts.append(f"GPA req {req_g}")
                req_info = " • ".join(req_parts) if req_parts else "No specific requirements given"
                st.write(f"{match_text} • {req_info}")
                if pct is not None:
                    st.progress(min(pct, 100.0) / 100.0)
                comment = row.get("Comment", "")
                with st.expander("💬 Comparison", expanded=False):
                    st.write(comment if comment else "No comment available.")
                st.divider()

    # Match tab
    with category_tabs[1]:
        if match_count == 0:
            st.write("No programs found in this category.")
        else:
            for _, row in filtered[filtered["Category"] == "Match"].iterrows():
                uni = row.get("University", "")
                prog = row.get("Programme", row.get("Program", ""))
                st.markdown(f"**{uni} – {prog}**")
                st.write("🥈 **Match**")
                pct = row["Percent Match"]
                req_a = row.get("A-level Requirements", None)
                req_g = row.get("GPA Requirements", None)
                match_text = "N/A" if pct is None else f"{pct:.1f}% Match"
                req_parts = []
                if isinstance(req_a, str) and req_a.strip():
                    req_parts.append(f"A-level req {req_a}")
                if pd.notna(req_g):
                    try:
                        req_val = float(req_g)
                        req_parts.append(f"GPA req {req_val:.2f}")
                    except:
                        req_parts.append(f"GPA req {req_g}")
                req_info = " • ".join(req_parts) if req_parts else "No specific requirements given"
                st.write(f"{match_text} • {req_info}")
                if pct is not None:
                    st.progress(min(pct, 100.0) / 100.0)
                comment = row.get("Comment", "")
                with st.expander("💬 Comparison", expanded=False):
                    st.write(comment if comment else "No comment available.")
                st.divider()

    # Reach tab
    with category_tabs[2]:
        if reach_count == 0:
            st.write("No programs found in this category.")
        else:
            for _, row in filtered[filtered["Category"] == "Reach"].iterrows():
                uni = row.get("University", "")
                prog = row.get("Programme", row.get("Program", ""))
                st.markdown(f"**{uni} – {prog}**")
                st.write("🚀 **Reach**")
                pct = row["Percent Match"]
                req_a = row.get("A-level Requirements", None)
                req_g = row.get("GPA Requirements", None)
                match_text = "N/A" if pct is None else f"{pct:.1f}% Match"
                req_parts = []
                if isinstance(req_a, str) and req_a.strip():
                    req_parts.append(f"A-level req {req_a}")
                if pd.notna(req_g):
                    try:
                        req_val = float(req_g)
                        req_parts.append(f"GPA req {req_val:.2f}")
                    except:
                        req_parts.append(f"GPA req {req_g}")
                req_info = " • ".join(req_parts) if req_parts else "No specific requirements given"
                st.write(f"{match_text} • {req_info}")
                if pct is not None:
                    st.progress(min(pct, 100.0) / 100.0)
                comment = row.get("Comment", "")
                with st.expander("💬 Comparison", expanded=False):
                    st.write(comment if comment else "No comment available.")
                st.divider()

    # N/A tab (if present)
    if na_count > 0:
        with category_tabs[3]:
            if na_count == 0:
                st.write("No programs found in this category.")
            else:
                for _, row in filtered[filtered["Category"] == "N/A"].iterrows():
                    uni = row.get("University", "")
                    prog = row.get("Programme", row.get("Program", ""))
                    st.markdown(f"**{uni} – {prog}**")
                    st.write("📓 **N/A (Not Categorized)**")
                    req_a = row.get("A-level Requirements", None)
                    req_g = row.get("GPA Requirements", None)
                    req_parts = []
                    if isinstance(req_a, str) and req_a.strip():
                        req_parts.append(f"A-level req {req_a}")
                    if pd.notna(req_g):
                        try:
                            req_val = float(req_g)
                            req_parts.append(f"GPA req {req_val:.2f}")
                        except:
                            req_parts.append(f"GPA req {req_g}")
                    req_info = " • ".join(req_parts) if req_parts else "No specific requirements given"
                    st.write(req_info)
                    comment = row.get("Comment", "")
                    with st.expander("💬 Comparison", expanded=False):
                        st.write(comment if comment else "No comment available.")
                    st.divider()

    # Save results for download step
    st.session_state.filtered_df = filtered
    # Proceed to next step
    st.write("----")
    st.write("When ready, proceed to download the full list of matches with comments.")
    if st.button("Next"):
        st.session_state.step = 4

def step_download():
    st.header("Step 4: Download Results")
    if "filtered_df" not in st.session_state or st.session_state.filtered_df.empty:
        st.error("No results to download. Please run the search first.")
        return
    output_df = st.session_state.filtered_df.copy()
    # Ensure only relevant columns (drop any index columns etc.)
    if "Unnamed: 0" in output_df.columns:
        output_df.drop(columns=["Unnamed: 0"], inplace=True)
    # Reorder columns to place Category and Comment at the end
    col_order = list(output_df.columns)
    for col in ["Category", "Comment"]:
        if col in col_order:
            col_order.remove(col)
            col_order.append(col)
    output_df = output_df[col_order]
    csv_data = output_df.to_csv(index=False)
    st.download_button("Download matched programmes as CSV", data=csv_data, file_name="matched_programmes.csv", mime="text/csv")
    st.success("CSV file is ready for download. Thank you for using the app!")

# Render the current step
if st.session_state.step == 1:
    step_grades()
elif st.session_state.step == 2:
    step_major()
elif st.session_state.step == 3:
    step_results()
elif st.session_state.step == 4:
    step_download()

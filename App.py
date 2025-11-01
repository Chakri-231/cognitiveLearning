# app.py
import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date as _date

# -----------------------
# App constants
# -----------------------
SUBJECTS = ["Math", "Science", "English", "Social Studies", "Computer Science", "Arts"]

REQUIRED_COLS = [
    "date","student_id","student_name","class_grade","section","subject",
    "exam_score","quiz_avg","assignments_completed_pct","attendance_pct",
    "quizzes_taken","last_assessment_days_ago","study_hours_per_week",
    "attention_score","participation_score","collaboration_score","conduct_incidents",
    "extracurricular_sports_hours","extracurricular_arts_hours","extracurricular_clubs_hours",
    "teacher_observation"
]

NUMERIC_COLS = [
    "exam_score","quiz_avg","assignments_completed_pct","attendance_pct",
    "quizzes_taken","last_assessment_days_ago","study_hours_per_week",
    "attention_score","participation_score","collaboration_score","conduct_incidents",
    "extracurricular_sports_hours","extracurricular_arts_hours","extracurricular_clubs_hours"
]

# -----------------------
# Cognitive model helpers
# -----------------------
def bayesian_mastery(row, prior=0.60, guess=0.20, slip=0.10):
    """
    Simple Bayesian updates with three binary evidences:
      1) exam_score >= 70
      2) quiz_avg >= 70
      3) assignments_completed_pct >= 80
    """
    p = prior
    evidences = [
        row["exam_score"] >= 70,
        row["quiz_avg"] >= 70,
        row["assignments_completed_pct"] >= 80
    ]
    for correct in evidences:
        if correct:
            num = p*(1 - slip)
            den = num + (1 - p)*guess
        else:
            num = p*slip
            den = num + (1 - p)*(1 - guess)
        if den > 0:
            p = num/den
    return float(np.clip(p, 0, 1))

def spaced_repetition_bucket(p_mastery, days_since):
    """
    Spacing guidance based on mastery and time since last assessment.
    """
    if p_mastery < 0.60 and days_since > 21:
        return "REVIEW TODAY"
    if p_mastery < 0.75 and days_since > 14:
        return "Review this week"
    if days_since > 21:
        return "Refresh this week"
    return "OK / Next week"

def risk_score(row, p_mastery, w_att=0.20, w_beh=0.25, w_mas=0.55):
    """
    Combine mastery, attendance, and behaviour into a 0-1 risk score.
    """
    mastery_risk = 1.0 - p_mastery
    att = row["attendance_pct"]
    att_risk = 0.0 if att >= 90 else (0.5 if att >= 80 else 1.0)
    beh_raw = (
        (3 - (row["attention_score"] - 2)) +
        (3 - (row["participation_score"] - 2)) +
        (3 - (row["collaboration_score"] - 2)) +
        (row["conduct_incidents"])
    )
    beh_risk = np.clip(beh_raw / 12.0, 0, 1)
    risk = np.clip(w_mas*mastery_risk + w_att*att_risk + w_beh*beh_risk, 0, 1)
    return float(risk)

def rule_based_feedback(row, p_mastery, risk):
    """
    Local 'AI-like' feedback without external APIs (replace with Gemini later).
    """
    msgs = []
    subj = row["subject"]
    if p_mastery < 0.6:
        msgs.append(f"Focus on {subj}: start with retrieval practice (2–3 short quizzes).")
    elif p_mastery < 0.75:
        msgs.append(f"{subj}: interleave practice with related topics; weekly spaced review.")
    else:
        msgs.append(f"{subj}: maintain momentum with fortnightly retrieval practice.")

    if row["attendance_pct"] < 80:
        msgs.append("Improve attendance; missing classes is impacting consolidation.")
    if row["conduct_incidents"] >= 3:
        msgs.append("Short behaviour coaching + clear class routines.")
    if row["attention_score"] <= 2:
        msgs.append("Try 10–15 min blocks + active recall to sustain attention.")
    if row["study_hours_per_week"] < 4:
        msgs.append("Add two short study blocks this week (30–40 min).")
    return " ".join(msgs)

# -----------------------
# Utility functions
# -----------------------
def empty_dataset():
    """
    Create an empty DataFrame with correct dtypes,
    esp. 'date' as datetime64[ns] so DateColumn editor works.
    """
    # Start with explicit dtypes
    base = {
        "date": pd.Series([], dtype="datetime64[ns]"),
        "student_id": pd.Series([], dtype="object"),
        "student_name": pd.Series([], dtype="object"),
        "class_grade": pd.Series([], dtype="object"),  # keep flexible
        "section": pd.Series([], dtype="object"),
        "subject": pd.Series([], dtype="object"),
        "teacher_observation": pd.Series([], dtype="object"),
    }
    # Numeric columns as float (safe with NaNs)
    for c in NUMERIC_COLS:
        base[c] = pd.Series([], dtype="float64")
    df = pd.DataFrame(base)[REQUIRED_COLS]
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure 'date' is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Numeric coercion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def validate_required(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return False
    return True

@st.cache_data
def load_csv(path_or_buffer):
    # Read, then coerce date safely (avoids parse_dates errors if column missing)
    df = pd.read_csv(path_or_buffer)
    if not validate_required(df):
        st.stop()
    df = coerce_types(df)
    df.dropna(subset=["student_id","subject"], inplace=True)
    return df

# -----------------------
# Page structure
# -----------------------
st.set_page_config(page_title="Minds at School - Faculty Entry + Analytics", layout="wide")
st.title("Minds at School: Faculty Data Entry & Learning Analytics")

# Session state for live entries
if "live_df" not in st.session_state:
    st.session_state.live_df = empty_dataset()

tab_entry, tab_analytics = st.tabs(["📝 Faculty Data Entry", "📊 Analytics & Modeling"])

# =======================
# Faculty Data Entry Tab
# =======================
with tab_entry:
    st.subheader("Add / Edit Observations")

    with st.form("faculty_entry_form", clear_on_submit=False):
        colA, colB, colC, colD = st.columns(4)
        with colA:
            entry_date = st.date_input("Date", value=_date.today())
            student_id = st.text_input("Student ID", placeholder="e.g., S101")
        with colB:
            student_name = st.text_input("Student Name", placeholder="e.g., Aarav Sharma")
            class_grade = st.selectbox("Class Grade", options=[6,7,8,9,10,11,12], index=4)
        with colC:
            section = st.selectbox("Section", options=["A","B","C","D"], index=0)
            subjects_selected = st.multiselect("Subjects (select one or many)", options=SUBJECTS, default=["Math"])
        with colD:
            teacher_observation = st.text_area(
                "Teacher Observation",
                placeholder="e.g., Focused in class; needs practice in algebra."
            )

        st.markdown("**Performance & Engagement**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            exam_score = st.number_input("Exam score (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
            quiz_avg = st.number_input("Quiz avg (0-100)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
            assignments_pct = st.number_input("Assignments completed % (0-100)", min_value=0.0, max_value=100.0, value=80.0, step=0.5)
        with col2:
            attendance_pct = st.number_input("Attendance % (0-100)", min_value=0.0, max_value=100.0, value=90.0, step=0.5)
            quizzes_taken = st.number_input("Quizzes taken (0-20)", min_value=0, max_value=20, value=6, step=1)
            last_ass_days = st.number_input("Days since last assessment (0-60)", min_value=0, max_value=60, value=10, step=1)
        with col3:
            study_hours = st.number_input("Study hours/week (0-40)", min_value=0.0, max_value=40.0, value=6.0, step=0.5)
            attention_score = st.number_input("Attention score (1-5)", min_value=1, max_value=5, value=3, step=1)
            participation_score = st.number_input("Participation score (1-5)", min_value=1, max_value=5, value=3, step=1)
        with col4:
            collaboration_score = st.number_input("Collaboration score (1-5)", min_value=1, max_value=5, value=3, step=1)
            conduct_incidents = st.number_input("Conduct incidents (0-5)", min_value=0, max_value=5, value=0, step=1)

        st.markdown("**Extra-curricular (hours/week)**")
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            sports_h = st.number_input("Sports hours (0-20)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
        with ec2:
            arts_h = st.number_input("Arts hours (0-20)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
        with ec3:
            clubs_h = st.number_input("Clubs hours (0-20)", min_value=0.0, max_value=20.0, value=0.5, step=0.5)

        add_rows = st.form_submit_button("➕ Add Row(s)")

        if add_rows:
            if not student_id or not student_name or not subjects_selected:
                st.warning("Please provide Student ID, Student Name, and select at least one Subject.")
            else:
                new_rows = []
                for subj in subjects_selected:
                    # FIX: Keep date as datetime, not string
                    new_rows.append({
                        "date": pd.to_datetime(entry_date),  # <-- important
                        "student_id": student_id.strip(),
                        "student_name": student_name.strip(),
                        "class_grade": class_grade,
                        "section": section,
                        "subject": subj,
                        "exam_score": exam_score,
                        "quiz_avg": quiz_avg,
                        "assignments_completed_pct": assignments_pct,
                        "attendance_pct": attendance_pct,
                        "quizzes_taken": quizzes_taken,
                        "last_assessment_days_ago": last_ass_days,
                        "study_hours_per_week": study_hours,
                        "attention_score": attention_score,
                        "participation_score": participation_score,
                        "collaboration_score": collaboration_score,
                        "conduct_incidents": conduct_incidents,
                        "extracurricular_sports_hours": sports_h,
                        "extracurricular_arts_hours": arts_h,
                        "extracurricular_clubs_hours": clubs_h,
                        "teacher_observation": teacher_observation.strip(),
                    })
                st.session_state.live_df = pd.concat(
                    [st.session_state.live_df, pd.DataFrame(new_rows)],
                    ignore_index=True
                )
                # Ensure types after append
                st.session_state.live_df = coerce_types(st.session_state.live_df)
                st.success(f"Added {len(new_rows)} row(s) for {student_name}.")

    st.divider()
    st.subheader("Live Dataset (Editable)")

    # Ensure correct dtype before editing to avoid Streamlit error
    st.session_state.live_df = coerce_types(st.session_state.live_df)

    if st.session_state.live_df.empty:
        st.info("No rows yet. Use the form above to add observations.")
    else:
        edited = st.data_editor(
            st.session_state.live_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                # FIX: DateColumn now compatible with datetime64[ns]
                "date": st.column_config.DateColumn("date", format="YYYY-MM-DD"),
                "exam_score": st.column_config.NumberColumn("exam_score", min_value=0, max_value=100, step=0.5),
                "quiz_avg": st.column_config.NumberColumn("quiz_avg", min_value=0, max_value=100, step=0.5),
                "assignments_completed_pct": st.column_config.NumberColumn("assignments_completed_pct", min_value=0, max_value=100, step=0.5),
                "attendance_pct": st.column_config.NumberColumn("attendance_pct", min_value=0, max_value=100, step=0.5),
                "quizzes_taken": st.column_config.NumberColumn("quizzes_taken", min_value=0, max_value=20, step=1),
                "last_assessment_days_ago": st.column_config.NumberColumn("last_assessment_days_ago", min_value=0, max_value=60, step=1),
                "study_hours_per_week": st.column_config.NumberColumn("study_hours_per_week", min_value=0, max_value=40, step=0.5),
                "attention_score": st.column_config.NumberColumn("attention_score", min_value=1, max_value=5, step=1),
                "participation_score": st.column_config.NumberColumn("participation_score", min_value=1, max_value=5, step=1),
                "collaboration_score": st.column_config.NumberColumn("collaboration_score", min_value=1, max_value=5, step=1),
                "conduct_incidents": st.column_config.NumberColumn("conduct_incidents", min_value=0, max_value=5, step=1),
                "extracurricular_sports_hours": st.column_config.NumberColumn("extracurricular_sports_hours", min_value=0, max_value=20, step=0.5),
                "extracurricular_arts_hours": st.column_config.NumberColumn("extracurricular_arts_hours", min_value=0, max_value=20, step=0.5),
                "extracurricular_clubs_hours": st.column_config.NumberColumn("extracurricular_clubs_hours", min_value=0, max_value=20, step=0.5),
            }
        )
        # Persist edits with correct dtypes
        st.session_state.live_df = coerce_types(edited)

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("🗑️ Clear all live entries"):
                st.session_state.live_df = empty_dataset()
                st.success("Cleared all live entries.")
        with c2:
            live_csv = st.session_state.live_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV (live dataset)", live_csv, file_name="faculty_entries.csv", mime="text/csv")
        with c3:
            st.info(f"Rows: {len(st.session_state.live_df)} | Students: {st.session_state.live_df['student_id'].nunique()}")

# =======================
# Analytics & Modeling Tab
# =======================
with tab_analytics:
    st.subheader("Choose Data Source for Analytics")
    src = st.radio("Data source", ["Uploaded CSV", "Sample CSV (bundled)", "Session (Faculty entries)"], horizontal=True)

    df = None
    if src == "Uploaded CSV":
        uploaded = st.file_uploader("Upload dataset CSV", type=["csv"], key="uploader_key")
        if uploaded:
            df = load_csv(uploaded)
        else:
            st.info("Upload a CSV to proceed.")
            st.stop()
    elif src == "Sample CSV (bundled)":
        try:
            df = load_csv("sample_student_learning_dataset.csv")
        except Exception as e:
            st.warning("Sample not found. Please place 'sample_student_learning_dataset.csv' next to app.py, or use another source.")
            st.stop()
    else:  # Session (Faculty entries)
        if st.session_state.live_df.empty:
            st.info("No live entries yet. Add rows in the Faculty Data Entry tab.")
            st.stop()
        df = st.session_state.live_df.copy()
        if not validate_required(df):
            st.stop()
        df = coerce_types(df)

    st.success(f"Loaded {len(df):,} rows | Students: {df['student_id'].nunique()} | Subjects: {df['subject'].nunique()}")

    st.sidebar.header("Modes")
    mode = st.sidebar.radio("Select mode", ["Manual (Human-in-the-loop)","Cognitive Model","AI (optional)"])

    st.sidebar.subheader("Manual Controls")
    thr_exam = st.sidebar.slider("Threshold: Exam score (pass)", 0, 100, 70)
    thr_att  = st.sidebar.slider("Threshold: Attendance (%)", 50, 100, 80)
    thr_beh  = st.sidebar.slider("Threshold: Conduct incidents (risk above)", 0, 5, 2)

    # Compute mastery/risk
    df["p_mastery"] = df.apply(bayesian_mastery, axis=1)
    df["risk"] = df.apply(lambda r: risk_score(r, r["p_mastery"]), axis=1)
    df["spacing"] = df.apply(lambda r: spaced_repetition_bucket(r["p_mastery"], r["last_assessment_days_ago"]), axis=1)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Exam", f"{df['exam_score'].mean():.1f}")
    c2.metric("Avg Quiz", f"{df['quiz_avg'].mean():.1f}")
    c3.metric("Avg Attendance", f"{df['attendance_pct'].mean():.1f}%")
    c4.metric("Avg Mastery", f"{df['p_mastery'].mean():.2f}")

    # Visuals
    st.subheader("Distributions by Subject")
    def hist_by_subject(metric):
        fig, ax = plt.subplots(figsize=(6,4))
        for subj in sorted(df['subject'].unique()):
            ax.hist(df[df['subject']==subj][metric], bins=15, alpha=0.4, label=subj)
        ax.set_title(f"{metric} distribution")
        ax.set_xlabel(metric); ax.set_ylabel("Count")
        ax.legend(fontsize=8, ncol=2)
        st.pyplot(fig)

    colA, colB = st.columns(2)
    with colA:
        hist_by_subject("exam_score")
    with colB:
        hist_by_subject("attendance_pct")

    st.subheader("Attendance vs Exam (color by Subject)")
    fig, ax = plt.subplots(figsize=(7,5))
    for subj in sorted(df['subject'].unique()):
        d = df[df['subject']==subj]
        ax.scatter(d["attendance_pct"], d["exam_score"], alpha=0.6, label=subj)
    ax.set_xlabel("Attendance %"); ax.set_ylabel("Exam score")
    ax.grid(alpha=0.2); ax.legend(fontsize=8, ncol=2)
    st.pyplot(fig)

    st.subheader("Correlation heatmap (numeric)")
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr().fillna(0)
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

    st.header("Student Drilldown")
    sel_student = st.selectbox("Choose a student", sorted(df["student_name"].unique()))
    sd = df[df["student_name"]==sel_student].sort_values("subject")

    c1, c2 = st.columns([1,1])
    with c1:
        st.write("Scores by Subject")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(sd["subject"], sd["exam_score"], color="#2e7d32")
        ax.set_ylim(0, 100); ax.set_ylabel("Exam score")
        ax.set_xticklabels(sd["subject"], rotation=30, ha="right")
        st.pyplot(fig)

    with c2:
        st.write("Estimated Mastery by Subject (Bayesian)")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(sd["subject"], sd["p_mastery"], color="#0277bd")
        ax.set_ylim(0, 1); ax.set_ylabel("P(Mastery)")
        ax.set_xticklabels(sd["subject"], rotation=30, ha="right")
        st.pyplot(fig)

    st.write("Spacing Recommendations")
    st.dataframe(sd[["subject","last_assessment_days_ago","p_mastery","spacing"]]
                 .rename(columns={
                     "last_assessment_days_ago":"days_since_assessment",
                     "p_mastery":"p_mastery(0-1)"
                 }))

    st.header("At-risk Students & Interventions")
    if mode == "Manual (Human-in-the-loop)":
        risky = df[
            (df["exam_score"] < thr_exam) |
            (df["attendance_pct"] < thr_att) |
            (df["conduct_incidents"] > thr_beh)
        ].copy()
    else:
        risky = df[df["risk"] >= 0.6].copy()

    risky = risky.sort_values(["risk","p_mastery","exam_score"], ascending=[False, True, True])
    st.write(f"Flagged rows: {len(risky)}")
    st.dataframe(risky[[
        "student_name","class_grade","section","subject",
        "exam_score","quiz_avg","attendance_pct","p_mastery","risk","spacing","teacher_observation"
    ]])

    st.subheader("Personalized Recommendations")
    recs = []
    for _, r in risky.head(20).iterrows():
        fb = rule_based_feedback(r, r["p_mastery"], r["risk"])
        recs.append({
            "student_name": r["student_name"],
            "subject": r["subject"],
            "p_mastery": round(r["p_mastery"], 2),
            "risk": round(r["risk"], 2),
            "recommendation": fb
        })
    st.dataframe(pd.DataFrame(recs))
    st.caption("Note: Replace the rule-based feedback with Gemini later for richer suggestions.")

    st.header("Export At-risk + Recommendations")
    teacher_note = st.text_area("Add a general note (included in export).")
    if st.button("Download current risky list + notes"):
        tmp = pd.DataFrame(recs)
        tmp["teacher_note"] = teacher_note
        csv = tmp.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="risky_students_with_recommendations.csv", mime="text/csv")
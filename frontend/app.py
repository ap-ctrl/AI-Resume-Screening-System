import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from backend.resume_analyzer import analyze_resume
from backend.resume_parser import extract_text_from_file
from backend.job_matcher import match_jobs
from backend.llm_explainer import explain_job_match


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening & Job Matching System",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "profile" not in st.session_state:
    st.session_state.profile = None

if "matched_jobs" not in st.session_state:
    st.session_state.matched_jobs = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_first_value(data, keys, default=""):
    """
    Return the first non-empty value from a dictionary.
    """

    if not isinstance(data, dict):
        return default

    for key in keys:

        value = data.get(key)

        if value is not None and str(value).strip():

            return value

    return default


def format_related_match(match):
    """
    Convert a related skill dictionary into readable text.
    """

    if isinstance(match, dict):

        required_skill = get_first_value(
            match,
            [
                "required_skill",
                "job_skill",
                "required",
                "skill"
            ],
            "Unknown skill"
        )

        resume_skills = get_first_value(
            match,
            [
                "resume_skills",
                "matched_skills",
                "candidate_skills"
            ],
            []
        )

        if isinstance(resume_skills, str):
            resume_skills = [resume_skills]

        if resume_skills:

            return (
                f"{required_skill} ← "
                + ", ".join(
                    map(str, resume_skills)
                )
            )

        return str(required_skill)

    return str(match)


def format_experience_item(item):
    """
    Convert experience object into readable text.
    """

    if not isinstance(item, dict):
        return str(item)

    role = get_first_value(
        item,
        [
            "role",
            "title",
            "position",
            "job_title",
            "designation"
        ]
    )

    company = get_first_value(
        item,
        [
            "company",
            "organization",
            "employer"
        ]
    )

    duration = get_first_value(
        item,
        [
            "duration",
            "period",
            "dates"
        ]
    )

    parts = []

    if role:
        parts.append(f"**{role}**")

    if company:
        parts.append(f"at {company}")

    if duration:
        parts.append(f"({duration})")

    if parts:
        return " ".join(parts)

    return str(item)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📄 AI Resume Screening & Job Matching System"
)

st.write(
    "Upload your resume or paste resume text to analyze "
    "your profile and find relevant job opportunities."
)


# ============================================================
# RESUME INPUT
# ============================================================

st.subheader("1️⃣ Resume Input")

input_method = st.radio(
    "Choose how you want to provide your resume:",
    [
        "Upload Resume",
        "Paste Resume Text"
    ],
    horizontal=True
)


# ============================================================
# FILE UPLOAD
# ============================================================

if input_method == "Upload Resume":

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=[
            "pdf",
            "docx",
            "txt"
        ]
    )

    if uploaded_file is not None:

        try:

            extracted_text = extract_text_from_file(
                uploaded_file
            )

            st.session_state.resume_text = extracted_text

            if extracted_text:

                st.success(
                    "Resume uploaded and text extracted successfully."
                )

        except Exception as e:

            st.error(
                f"Error reading resume: {e}"
            )


# ============================================================
# PASTE RESUME
# ============================================================

else:

    pasted_text = st.text_area(
        "Paste your resume text below:",
        height=300,
        placeholder="Paste your complete resume here..."
    )

    if pasted_text.strip():

        st.session_state.resume_text = pasted_text


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🚀 Analyze Resume",
    type="primary"
):

    resume_text = st.session_state.resume_text

    if not resume_text or not resume_text.strip():

        st.warning(
            "Please upload a resume or paste resume text first."
        )

        st.stop()


    # ========================================================
    # RESUME ANALYSIS
    # ========================================================

    with st.spinner(
        "Analyzing your resume..."
    ):

        try:

            profile = analyze_resume(
                resume_text
            )

            st.session_state.profile = profile

        except Exception as e:

            st.error(
                f"Resume analysis failed: {e}"
            )

            st.stop()


    # ========================================================
    # JOB MATCHING
    # ========================================================

    with st.spinner(
        "Finding the most relevant jobs..."
    ):

        try:

            matched_jobs = match_jobs(
                resume_text,
                top_n=5,
                resume_profile=profile
            )

            st.session_state.matched_jobs = matched_jobs

        except Exception as e:

            st.error(
                f"Job matching failed: {e}"
            )

            st.stop()


# ============================================================
# DISPLAY PROFILE
# ============================================================

if st.session_state.profile is not None:

    profile = st.session_state.profile

    st.subheader(
        "2️⃣ Resume Profile"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = profile.get(
        "summary",
        ""
    )

    if summary:

        st.write(
            "### 📝 Summary"
        )

        st.write(
            summary
        )


    # ========================================================
    # SKILLS
    # ========================================================

    technical_skills = profile.get(
        "technical_skills",
        []
    )

    soft_skills = profile.get(
        "soft_skills",
        []
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "### 💻 Technical Skills"
        )

        if technical_skills:

            st.write(
                ", ".join(
                    map(str, technical_skills)
                )
            )

        else:

            st.info(
                "No technical skills detected."
            )


    with col2:

        st.write(
            "### 🤝 Soft Skills"
        )

        if soft_skills:

            st.write(
                ", ".join(
                    map(str, soft_skills)
                )
            )

        else:

            st.info(
                "No soft skills detected."
            )


    # ========================================================
    # EXPERIENCE
    # ========================================================

    experience = profile.get(
        "experience",
        []
    )

    if experience:

        st.write(
            "### 💼 Experience"
        )

        for item in experience:

            st.write(
                f"- {format_experience_item(item)}"
            )


    # ========================================================
    # PROJECTS
    # ========================================================

    projects = profile.get(
        "projects",
        []
    )

    if projects:

        st.write(
            "### 🚀 Projects"
        )

        for project in projects:

            if isinstance(project, dict):

                name = get_first_value(
                    project,
                    [
                        "name",
                        "title",
                        "project_name"
                    ],
                    "Project"
                )

                description = get_first_value(
                    project,
                    [
                        "description",
                        "details",
                        "summary"
                    ]
                )

                technologies = get_first_value(
                    project,
                    [
                        "technologies",
                        "tech_stack",
                        "tools"
                    ],
                    []
                )

                st.write(
                    f"**{name}**"
                )

                if description:

                    st.write(
                        description
                    )

                if technologies:

                    if isinstance(
                        technologies,
                        str
                    ):

                        technologies = [
                            technologies
                        ]

                    st.write(
                        "Technologies: "
                        + ", ".join(
                            map(str, technologies)
                        )
                    )

            else:

                st.write(
                    f"- {project}"
                )


# ============================================================
# JOB MATCHING
# ============================================================

if st.session_state.matched_jobs is not None:

    matched_jobs = st.session_state.matched_jobs

    st.subheader(
        "3️⃣ Job Matching"
    )

    if matched_jobs.empty:

        st.warning(
            "No matching jobs were found."
        )

        st.stop()


    st.success(
        f"Found {len(matched_jobs)} relevant job matches."
    )


    # ========================================================
    # JOB RESULTS
    # ========================================================

    for index, (_, job) in enumerate(
        matched_jobs.iterrows()
    ):

        st.markdown(
            "---"
        )


        # ====================================================
        # BASIC INFORMATION
        # ====================================================

        rank = job.get(
            "Rank",
            index + 1
        )

        job_title = job.get(
            "Job_Title",
            "Unknown Job"
        )

        category = job.get(
            "Category",
            ""
        )


        st.write(
            f"### #{rank} {job_title}"
        )

        if category:

            st.caption(
                f"Category: {category}"
            )


        # ====================================================
        # SCORES
        # ====================================================

        score = float(
            job.get(
                "Similarity",
                0
            )
        )

        tfidf_score = float(
            job.get(
                "TFIDF_Similarity",
                0
            )
        )

        semantic_score = float(
            job.get(
                "Semantic_Similarity",
                0
            )
        )

        required_score = float(
            job.get(
                "Required_Skill_Coverage",
                0
            )
        )

        preferred_score = float(
            job.get(
                "Preferred_Skill_Coverage",
                0
            )
        )


        # ====================================================
        # FINAL SCORE
        # ====================================================

        st.metric(
            "Final Match Score",
            f"{score * 100:.2f}%"
        )


        # ====================================================
        # SCORE BREAKDOWN
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "TF-IDF",
                f"{tfidf_score * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Semantic",
                f"{semantic_score * 100:.2f}%"
            )


        with col3:

            st.metric(
                "Required Skills",
                f"{required_score * 100:.2f}%"
            )


        with col4:

            st.metric(
                "Preferred Skills",
                f"{preferred_score * 100:.2f}%"
            )


        # ====================================================
        # EXACT MATCHES
        # ====================================================

        exact_matches = job.get(
            "Exact_Matches",
            []
        )

        if exact_matches:

            st.write(
                "### ✅ Exact Skill Matches"
            )

            for skill in exact_matches:

                st.write(
                    f"- {skill}"
                )


        # ====================================================
        # RELATED MATCHES
        # ====================================================

        related_matches = job.get(
            "Related_Matches",
            []
        )

        if related_matches:

            st.write(
                "### 🟡 Related Skill Matches"
            )

            for match in related_matches:

                st.write(
                    f"- {format_related_match(match)}"
                )


        # ====================================================
        # MISSING SKILLS
        # ====================================================

        missing_skills = job.get(
            "Missing_Skills",
            []
        )

        if missing_skills:

            st.write(
                "### ❌ Missing Required Skills"
            )

            for skill in missing_skills:

                st.write(
                    f"- {skill}"
                )


        # ====================================================
        # MISSING PREFERRED SKILLS
        # ====================================================

        missing_preferred = job.get(
            "Missing_Preferred_Skills",
            []
        )

        if missing_preferred:

            st.write(
                "### ⚪ Missing Preferred Skills"
            )

            for skill in missing_preferred:

                st.write(
                    f"- {skill}"
                )


        # ====================================================
        # AI EXPLANATION
        # ====================================================

        st.write(
            "### 🤖 AI Explanation"
        )

        explanation_key = (
            f"explain_job_{index}"
        )

        if st.button(
            "Why does this job match?",
            key=explanation_key
        ):

            with st.spinner(
                "Generating grounded explanation..."
            ):

                try:

                    explanation = explain_job_match(
                        job
                    )

                    st.markdown(
                        explanation
                    )

                except Exception as e:

                    st.error(
                        f"Could not generate explanation: {e}"
                    )


        # ====================================================
        # JOB DESCRIPTION
        # ====================================================

        job_description = job.get(
            "Job_Description",
            ""
        )

        if job_description:

            with st.expander(
                "📋 View Job Description"
            ):

                st.write(
                    job_description
                )
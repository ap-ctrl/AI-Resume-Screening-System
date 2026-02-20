import streamlit as st
from backend.predict import predict_resume
from backend.job_matcher import match_jobs

st.set_page_config(page_title="AI Resume Screening System", layout="centered")

st.title("AI Resume Screening & Job Matching System")

st.write("Paste resume text below to find the best matching jobs based on skill similarity.")

resume_text = st.text_area("Resume Text", height=300)

if st.button("Analyze Resume"):

    if resume_text.strip() == "":
        st.warning("Please enter resume text.")
    else:

        # =============================
        # 1️⃣ JOB MATCHING (PRIMARY)
        # =============================
        st.subheader("Top Job Matches For Your Resume")

        top_jobs = match_jobs(resume_text)

        for index, row in top_jobs.iterrows():
            match_score = round(row["Similarity"] * 100, 2)

            st.markdown(f"### {row['Job_Title']}")
            st.write(f"Category: {row['Category']}")
            st.write(f"Match Score: {match_score}%")
            st.progress(match_score / 100)
            
            # 🔴 Weak Match Warning
        if match_score < 25:
         st.warning("Low confidence match – resume may not strongly align with this role.")

         st.write("---")

        # =============================
        # 2️⃣ OPTIONAL CATEGORY (SECONDARY)
        # =============================
        predicted_category = predict_resume(resume_text)

        st.subheader("Predicted Resume Domain (Secondary Insight)")
        st.info(predicted_category)
# import streamlit as st
# from backend.predict import predict_resume

# # Page config
# st.set_page_config(page_title="AI Resume Classifier", layout="centered")

# st.title("AI Resume Classification System")
# st.write("Paste resume text below to predict the job category.")

# # Text input
# resume_text = st.text_area("Resume Text", height=300)

# if st.button("Predict Category"):
#     if resume_text.strip() == "":
#         st.warning("Please enter resume text.")
#     else:
#         result = predict_resume(resume_text)
#         st.success(f"Predicted Category: {result}") 

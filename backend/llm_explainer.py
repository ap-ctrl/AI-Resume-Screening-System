import json

from backend.groq_client import get_llm_response


# ============================================================
# BUILD GROUNDED EXPLANATION PROMPT
# ============================================================

def build_job_explanation_prompt(
    job_title,
    category,
    match_score,
    tfidf_score,
    semantic_score,
    required_skill_coverage,
    preferred_skill_coverage,
    exact_matches,
    related_matches,
    missing_skills,
    missing_preferred_skills,
    job_description
):
    """
    Build a grounded prompt for explaining why a resume
    matches a particular job.

    IMPORTANT:
    The LLM does NOT calculate the match score.
    It only explains the evidence supplied by Python.
    """

    prompt = f"""
You are an AI career assistant explaining a resume-to-job match.

Your task is ONLY to explain the matching evidence provided below.

IMPORTANT RULES:

1. Do NOT calculate or change the match score.
2. Do NOT invent skills that are not listed.
3. Do NOT claim the candidate has experience that is not provided.
4. Do NOT claim the candidate is guaranteed to get the job.
5. Use only the provided exact matches, related matches, missing skills,
   scores, and job description.
6. Clearly distinguish exact matches from related matches.
7. Be concise and professional.
8. Do not exaggerate the candidate's qualifications.

JOB INFORMATION
---------------
Job Title:
{job_title}

Category:
{category}

Match Score:
{match_score:.2%}

TF-IDF Similarity:
{tfidf_score:.2%}

Semantic Similarity:
{semantic_score:.2%}

Required Skill Coverage:
{required_skill_coverage:.2%}

Preferred Skill Coverage:
{preferred_skill_coverage:.2%}


EXACT SKILL MATCHES
-------------------
{json.dumps(exact_matches, indent=2)}


RELATED SKILL MATCHES
--------------------
{json.dumps(related_matches, indent=2)}


MISSING REQUIRED SKILLS
-----------------------
{json.dumps(missing_skills, indent=2)}


MISSING PREFERRED SKILLS
------------------------
{json.dumps(missing_preferred_skills, indent=2)}


JOB DESCRIPTION
---------------
{job_description}


OUTPUT FORMAT
-------------

Use exactly these sections:

Why this job matches:
Write 2-4 sentences explaining the strongest evidence.

Strong matches:
Use short bullet points.

Related evidence:
Use short bullet points. If there are no related matches, write:
None identified.

Main skill gaps:
Use short bullet points. If there are no missing skills, write:
None identified.

Keep the explanation grounded in the supplied data.
"""

    return prompt


# ============================================================
# EXPLAIN JOB MATCH
# ============================================================

def explain_job_match(job):
    """
    Generate a grounded natural-language explanation
    for one matched job.

    `job` can be a pandas Series or dictionary.
    """

    # --------------------------------------------------------
    # Convert pandas Series to dictionary if necessary
    # --------------------------------------------------------

    if hasattr(job, "to_dict"):
        job = job.to_dict()

    if not isinstance(job, dict):
        raise TypeError(
            "Job must be a dictionary or pandas Series."
        )


    # --------------------------------------------------------
    # Extract values safely
    # --------------------------------------------------------

    job_title = job.get(
        "Job_Title",
        "Unknown Job"
    )

    category = job.get(
        "Category",
        ""
    )

    match_score = float(
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

    required_skill_coverage = float(
        job.get(
            "Required_Skill_Coverage",
            0
        )
    )

    preferred_skill_coverage = float(
        job.get(
            "Preferred_Skill_Coverage",
            0
        )
    )

    exact_matches = job.get(
        "Exact_Matches",
        []
    )

    related_matches = job.get(
        "Related_Matches",
        []
    )

    missing_skills = job.get(
        "Missing_Skills",
        []
    )

    missing_preferred_skills = job.get(
        "Missing_Preferred_Skills",
        []
    )

    job_description = job.get(
        "Job_Description",
        ""
    )


    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    if exact_matches is None:
        exact_matches = []

    if related_matches is None:
        related_matches = []

    if missing_skills is None:
        missing_skills = []

    if missing_preferred_skills is None:
        missing_preferred_skills = []

    if job_description is None:
        job_description = ""


    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = build_job_explanation_prompt(
        job_title=job_title,
        category=category,
        match_score=match_score,
        tfidf_score=tfidf_score,
        semantic_score=semantic_score,
        required_skill_coverage=required_skill_coverage,
        preferred_skill_coverage=preferred_skill_coverage,
        exact_matches=exact_matches,
        related_matches=related_matches,
        missing_skills=missing_skills,
        missing_preferred_skills=missing_preferred_skills,
        job_description=job_description
    )


    # --------------------------------------------------------
    # Call Groq
    # --------------------------------------------------------

    response = get_llm_response(
        prompt
    )


    if response is None:
        raise ValueError(
            "Groq returned an empty response."
        )


    response = str(response).strip()


    if not response:
        raise ValueError(
            "Groq returned an empty explanation."
        )


    return response


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("JOB EXPLANATION TEST")
    print("=" * 60)

    sample_job = {

        "Job_Title": "Software Engineer",

        "Category": "INFORMATION-TECHNOLOGY",

        "Similarity": 0.5010,

        "TFIDF_Similarity": 0.1779,

        "Semantic_Similarity": 0.6848,

        "Required_Skill_Coverage": 0.65,

        "Preferred_Skill_Coverage": 0.0,

        "Exact_Matches": [
            "Data Structures & Algorithms",
            "Java",
            "Python",
            "REST API",
            "SQL",
            "System Design"
        ],

        "Related_Matches": [
            {
                "required_skill": "Cloud Deployment",
                "resume_skills": ["Docker"]
            }
        ],

        "Missing_Skills": [
            "Backend Development",
            "debugging",
            "scalable application development"
        ],

        "Missing_Preferred_Skills": [],

        "Job_Description": (
            "Looking for a software engineer skilled in Python, "
            "Java, data structures, algorithms, backend development, "
            "REST APIs, SQL databases, cloud deployment, system design, "
            "debugging, and scalable application development."
        )
    }


    try:

        explanation = explain_job_match(
            sample_job
        )

        print("\nAI EXPLANATION:\n")
        print(explanation)

    except Exception as e:

        print("\nERROR:")
        print(e)
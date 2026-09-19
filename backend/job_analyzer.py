import json

from backend.groq_client import get_llm_response
from backend.skill_normalizer import normalize_skills


# ============================================================
# JOB ANALYSIS PROMPT
# ============================================================

JOB_ANALYSIS_PROMPT = """
You are an expert job-description analysis system.

Your task is to extract structured requirements
from the provided job description.

IMPORTANT RULES:

1. Only extract information explicitly present
   in the job description.

2. Do NOT invent skills, qualifications,
   experience or education requirements.

3. Separate required skills from preferred skills.

4. Required skills are skills that the job clearly
   expects or requires from the candidate.

5. Preferred skills are skills described as:
   preferred, desirable, nice-to-have, bonus,
   plus, or similar.

6. If the job does not specify preferred skills,
   return an empty list.

7. If experience is not specified,
   return an empty string.

8. If education is not specified,
   return an empty string.

9. Return only information supported by the
   job description.

10. Do not add explanations.

Analyze this job:

---------------- JOB START ----------------

__JOB_TEXT__

----------------- JOB END -----------------
"""


# ============================================================
# JSON SCHEMA
# ============================================================

JOB_SCHEMA = {

    "type": "object",

    "properties": {

        "required_skills": {

            "type": "array",

            "items": {
                "type": "string"
            }
        },

        "preferred_skills": {

            "type": "array",

            "items": {
                "type": "string"
            }
        },

        "experience": {

            "type": "string"
        },

        "education": {

            "type": "string"
        }
    },

    "required": [
        "required_skills",
        "preferred_skills",
        "experience",
        "education"
    ],

    "additionalProperties": False
}


# ============================================================
# EMPTY JOB PROFILE
# ============================================================

def empty_job_profile():

    return {

        "required_skills": [],

        "preferred_skills": [],

        "experience": "",

        "education": ""
    }


# ============================================================
# VALIDATE JOB PROFILE
# ============================================================

def validate_job_profile(
    profile
):
    """
    Make sure the job profile contains
    all required fields.
    """

    default_profile = (
        empty_job_profile()
    )


    if not isinstance(
        profile,
        dict
    ):

        return default_profile


    for key in default_profile:

        if key not in profile:

            profile[key] = (
                default_profile[key]
            )


    return profile


# ============================================================
# ANALYZE JOB
# ============================================================

def analyze_job(
    job_text
):
    """
    Analyze a job description using Groq.

    Returns:
        Structured job profile.
    """

    if not job_text:

        raise ValueError(
            "Job description cannot be empty."
        )


    if not job_text.strip():

        raise ValueError(
            "Job description cannot be empty."
        )


    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = JOB_ANALYSIS_PROMPT.replace(
        "__JOB_TEXT__",
        job_text
    )


    # --------------------------------------------------------
    # Structured output configuration
    # --------------------------------------------------------

    response_format = {

        "type": "json_schema",

        "json_schema": {

            "name": "job_profile",

            "strict": True,

            "schema": JOB_SCHEMA
        }
    }


    # --------------------------------------------------------
    # Call Groq
    # --------------------------------------------------------

    raw_response = get_llm_response(

        prompt=prompt,

        temperature=0.1,

        max_tokens=3000,

        response_format=response_format
    )


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        profile = json.loads(
            raw_response
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            "Groq returned invalid JSON "
            "for the job description.\n\n"
            f"Raw response:\n{raw_response}"
        ) from error


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    profile = validate_job_profile(
        profile
    )


    # --------------------------------------------------------
    # Normalize skills
    # --------------------------------------------------------

    profile[
        "required_skills"
    ] = normalize_skills(
        profile.get(
            "required_skills",
            []
        )
    )


    profile[
        "preferred_skills"
    ] = normalize_skills(
        profile.get(
            "preferred_skills",
            []
        )
    )


    return profile


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    sample_job = """
    Data Analyst

    We are looking for a Data Analyst who can work
    with SQL, Python, Excel, Power BI and Tableau.

    The candidate should have experience with
    data cleaning, data analysis and reporting.

    Knowledge of statistical analysis and
    business intelligence is preferred.

    A Bachelor's degree in Computer Science,
    Statistics, Mathematics or a related field
    is preferred.
    """


    result = analyze_job(
        sample_job
    )


    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )
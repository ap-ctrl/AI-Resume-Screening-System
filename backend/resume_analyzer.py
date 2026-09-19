import json

from backend.groq_client import get_llm_response


# ============================================================
# RESUME ANALYSIS PROMPT
# ============================================================

RESUME_ANALYSIS_PROMPT = """
You are an expert resume information extraction system.

Analyze the provided resume and extract structured information.

IMPORTANT RULES:

1. Only extract information explicitly present in the resume.

2. Never invent:
   - skills
   - companies
   - job titles
   - degrees
   - certifications
   - experience
   - projects
   - achievements

3. If a piece of information is not present,
   use an empty string or an empty array.

4. Preserve the actual meaning of the resume.

5. Technical skills must contain actual technical,
   analytical, software, programming, database,
   machine-learning, AI, cloud, or development skills.

6. Soft skills should contain interpersonal or
   behavioral skills explicitly mentioned or clearly
   demonstrated in the resume.

7. Keep project technologies separate from the
   project description.

8. Return the information according to the supplied
   JSON schema.

9. Do not invent a summary.

10. The output must contain ONLY structured JSON.

Resume:

---------------- RESUME START ----------------

__RESUME_TEXT__

----------------- RESUME END -----------------
"""


# ============================================================
# JSON SCHEMA
# ============================================================

RESUME_SCHEMA = {

    "type": "object",

    "properties": {

        "summary": {
            "type": "string"
        },

        "technical_skills": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "soft_skills": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "education": {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "degree": {
                        "type": "string"
                    },

                    "institution": {
                        "type": "string"
                    },

                    "duration": {
                        "type": "string"
                    }
                },

                "required": [
                    "degree",
                    "institution",
                    "duration"
                ],

                "additionalProperties": False
            }
        },

        "experience": {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "job_title": {
                        "type": "string"
                    },

                    "company": {
                        "type": "string"
                    },

                    "duration": {
                        "type": "string"
                    },

                    "description": {
                        "type": "string"
                    }
                },

                "required": [
                    "job_title",
                    "company",
                    "duration",
                    "description"
                ],

                "additionalProperties": False
            }
        },

        "projects": {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "name": {
                        "type": "string"
                    },

                    "description": {
                        "type": "string"
                    },

                    "technologies": {

                        "type": "array",

                        "items": {
                            "type": "string"
                        }
                    }
                },

                "required": [
                    "name",
                    "description",
                    "technologies"
                ],

                "additionalProperties": False
            }
        },

        "certifications": {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "name": {
                        "type": "string"
                    },

                    "issuer": {
                        "type": "string"
                    },

                    "year": {
                        "type": "string"
                    }
                },

                "required": [
                    "name",
                    "issuer",
                    "year"
                ],

                "additionalProperties": False
            }
        },

        "languages": {

            "type": "array",

            "items": {
                "type": "string"
            }
        }
    },

    "required": [
        "summary",
        "technical_skills",
        "soft_skills",
        "education",
        "experience",
        "projects",
        "certifications",
        "languages"
    ],

    "additionalProperties": False
}


# ============================================================
# EMPTY PROFILE
# ============================================================

def empty_resume_profile():

    return {

        "summary": "",

        "technical_skills": [],

        "soft_skills": [],

        "education": [],

        "experience": [],

        "projects": [],

        "certifications": [],

        "languages": []
    }


# ============================================================
# VALIDATE PROFILE
# ============================================================

def validate_resume_profile(
    profile
):
    """
    Make sure the returned profile has
    the expected top-level fields.
    """

    default_profile = (
        empty_resume_profile()
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
# ANALYZE RESUME
# ============================================================

def analyze_resume(
    resume_text
):
    """
    Analyze resume using Groq structured outputs.

    Returns:
        Python dictionary containing structured
        resume information.
    """

    if not resume_text:

        raise ValueError(
            "Resume text cannot be empty."
        )


    if not resume_text.strip():

        raise ValueError(
            "Resume text cannot be empty."
        )


    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = RESUME_ANALYSIS_PROMPT.replace(
        "__RESUME_TEXT__",
        resume_text
    )


    # --------------------------------------------------------
    # Structured Output configuration
    # --------------------------------------------------------

    response_format = {

        "type": "json_schema",

        "json_schema": {

            "name": "resume_profile",

            "strict": True,

            "schema": RESUME_SCHEMA
        }
    }


    # --------------------------------------------------------
    # Call Groq
    # --------------------------------------------------------

    raw_response = get_llm_response(

        prompt=prompt,

        temperature=0.1,

        max_tokens=8000,

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
            "Groq returned invalid JSON even though "
            "structured output was requested.\n\n"
            f"Raw response:\n{raw_response}"
        ) from error


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    return validate_resume_profile(
        profile
    )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    sample_resume = """

    John Doe

    B.Tech Computer Science Engineering

    Skills:

    Python,
    Java,
    SQL,
    Machine Learning,
    Pandas

    Projects:

    AI Resume Screening System using
    Python, Streamlit and Scikit-learn.

    Education:

    B.Tech in Computer Science Engineering

    """

    result = analyze_resume(
        sample_resume
    )


    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )


# import json
# import re

# from backend.groq_client import get_llm_response


# # ============================================================
# # RESUME ANALYSIS PROMPT
# # ============================================================

# RESUME_ANALYSIS_PROMPT = """
# You are an expert resume information extraction system.

# Your task is to analyze the provided resume and extract
# structured information from it.

# IMPORTANT RULES:

# 1. Only extract information that is explicitly present
#    or clearly supported by the resume.

# 2. Do NOT invent:
#    - skills
#    - companies
#    - job titles
#    - degrees
#    - certifications
#    - years of experience
#    - projects
#    - achievements

# 3. If information is not present, return an empty list.

# 4. Keep technical skills as specific as possible.

# 5. Separate technical skills from soft skills.

# 6. Preserve the actual meaning of the resume.

# 7. Return ONLY valid JSON.

# 8. Do not add markdown.
# 9. Do not add explanations outside the JSON.

# Use exactly this JSON structure:

# {
#     "summary": "",
#     "technical_skills": [],
#     "soft_skills": [],
#     "education": [],
#     "experience": [],
#     "projects": [],
#     "certifications": [],
#     "languages": []
# }

# For experience, include objects using this structure:

# {
#     "job_title": "",
#     "company": "",
#     "duration": "",
#     "description": ""
# }

# For education, include objects using this structure:

# {
#     "degree": "",
#     "institution": "",
#     "duration": ""
# }

# For projects, include objects using this structure:

# {
#     "name": "",
#     "description": "",
#     "technologies": []
# }

# For certifications, include objects using this structure:

# {
#     "name": "",
#     "issuer": "",
#     "year": ""
# }

# Now analyze the following resume:

# ---------------- RESUME START ----------------

# __RESUME_TEXT__

# ----------------- RESUME END -----------------
# """


# # ============================================================
# # CLEAN JSON RESPONSE
# # ============================================================

# def clean_json_response(response):
#     """
#     Clean common formatting issues from an LLM JSON response.
#     """

#     response = response.strip()

#     # Remove ```json from the beginning
#     response = re.sub(
#         r"^```json\s*",
#         "",
#         response,
#         flags=re.IGNORECASE
#     )

#     # Remove ``` from the beginning
#     response = re.sub(
#         r"^```\s*",
#         "",
#         response
#     )

#     # Remove ``` from the end
#     response = re.sub(
#         r"\s*```$",
#         "",
#         response
#     )

#     return response.strip()


# # ============================================================
# # DEFAULT EMPTY PROFILE
# # ============================================================

# def empty_resume_profile():
#     """
#     Return a safe empty resume profile.
#     """

#     return {
#         "summary": "",
#         "technical_skills": [],
#         "soft_skills": [],
#         "education": [],
#         "experience": [],
#         "projects": [],
#         "certifications": [],
#         "languages": []
#     }


# # ============================================================
# # VALIDATE PROFILE
# # ============================================================

# def validate_resume_profile(profile):
#     """
#     Make sure the returned profile contains all required fields.
#     """

#     default_profile = empty_resume_profile()

#     if not isinstance(profile, dict):
#         return default_profile

#     for key in default_profile:

#         if key not in profile:
#             profile[key] = default_profile[key]

#     # Make sure summary is a string
#     if not isinstance(
#         profile["summary"],
#         str
#     ):
#         profile["summary"] = ""

#     # Make sure list fields are actually lists
#     list_fields = [
#         "technical_skills",
#         "soft_skills",
#         "education",
#         "experience",
#         "projects",
#         "certifications",
#         "languages"
#     ]

#     for field in list_fields:

#         if not isinstance(
#             profile[field],
#             list
#         ):
#             profile[field] = []

#     return profile


# # ============================================================
# # ANALYZE RESUME
# # ============================================================

# def analyze_resume(resume_text):
#     """
#     Analyze resume text using Groq.

#     Parameters:
#         resume_text: Raw resume text.

#     Returns:
#         Structured resume profile as a Python dictionary.
#     """

#     if not resume_text or not resume_text.strip():

#         raise ValueError(
#             "Resume text cannot be empty."
#         )

#     # --------------------------------------------------------
#     # IMPORTANT:
#     # We use .replace() instead of .format()
#     # because the prompt contains JSON braces.
#     # --------------------------------------------------------

#     prompt = RESUME_ANALYSIS_PROMPT.replace(
#         "__RESUME_TEXT__",
#         resume_text
#     )

#     # --------------------------------------------------------
#     # Send prompt to Groq
#     # --------------------------------------------------------

#     raw_response = get_llm_response(
#         prompt=prompt,
#         temperature=0.1,
#         max_tokens=2500
#     )

#     # --------------------------------------------------------
#     # Clean response
#     # --------------------------------------------------------

#     cleaned_response = clean_json_response(
#         raw_response
#     )

#     # --------------------------------------------------------
#     # Convert JSON string into Python dictionary
#     # --------------------------------------------------------

#     try:

#         profile = json.loads(
#             cleaned_response
#         )

#     except json.JSONDecodeError as error:

#         raise ValueError(
#             "Groq returned an invalid JSON response.\n\n"
#             f"Raw response:\n{raw_response}"
#         ) from error

#     # --------------------------------------------------------
#     # Validate structure
#     # --------------------------------------------------------

#     return validate_resume_profile(
#         profile
#     )


# # ============================================================
# # MANUAL TEST
# # ============================================================

# if __name__ == "__main__":

#     sample_resume = """
#     John Doe

#     B.Tech Computer Science Engineering

#     Skills:
#     Python, Java, SQL, Machine Learning, Pandas

#     Projects:
#     AI Resume Screening System using Python,
#     Streamlit and Scikit-learn.

#     Education:
#     B.Tech in Computer Science Engineering
#     """

#     result = analyze_resume(
#         sample_resume
#     )

#     print(
#         json.dumps(
#             result,
#             indent=4
#         )
#     )
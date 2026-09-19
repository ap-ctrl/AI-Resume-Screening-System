import os
from functools import lru_cache

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from backend.job_profile_cache import get_job_profiles
from backend.skill_matcher import analyze_job_skill_match
from backend.skill_normalizer import (
    extract_known_skills,
    normalize_skills
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)


# ============================================================
# MATCHING CONFIGURATION
# ============================================================

DEFAULT_TOP_N = 5


TFIDF_WEIGHT = 0.20

SEMANTIC_WEIGHT = 0.30

REQUIRED_SKILL_WEIGHT = 0.40

PREFERRED_SKILL_WEIGHT = 0.10


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

tfidf = joblib.load(
    VECTORIZER_PATH
)


# ============================================================
# LOAD SEMANTIC MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():

    print(
        "\nLoading semantic embedding model..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print(
        "Semantic embedding model loaded."
    )

    return model


# ============================================================
# LOAD JOB PROFILES
# ============================================================

@lru_cache(maxsize=1)
def get_cached_jobs():

    profiles = get_job_profiles()


    if not profiles:

        raise ValueError(
            "No job profiles were found."
        )


    return profiles


# ============================================================
# PREPARE JOB DATAFRAME
# ============================================================

def get_jobs_dataframe():

    profiles = get_cached_jobs()


    jobs_df = pd.DataFrame(
        profiles
    )


    if "Job_Description" not in jobs_df.columns:

        raise ValueError(
            "Job profiles are missing "
            "'Job_Description'."
        )


    jobs_df = jobs_df.dropna(
        subset=[
            "Job_Description"
        ]
    ).copy()


    return jobs_df


# ============================================================
# PREPARE RESUME SKILLS
# ============================================================

def prepare_resume_skills(
    resume_text,
    resume_profile=None
):
    """
    Get resume skills from the structured
    Groq resume profile.

    We also extract known skills directly
    from the resume text as a deterministic
    backup.

    This gives us:

        LLM extraction
              +
        deterministic extraction
    """

    skills = []


    # --------------------------------------------------------
    # Skills from Groq structured profile
    # --------------------------------------------------------

    if isinstance(
        resume_profile,
        dict
    ):

        technical_skills = (
            resume_profile.get(
                "technical_skills",
                []
            )
        )


        skills.extend(
            technical_skills
        )


        projects = (
            resume_profile.get(
                "projects",
                []
            )
        )


        for project in projects:

            if not isinstance(
                project,
                dict
            ):

                continue


            technologies = (
                project.get(
                    "technologies",
                    []
                )
            )


            skills.extend(
                technologies
            )


    # --------------------------------------------------------
    # Known skills directly from resume text
    # --------------------------------------------------------

    extracted_skills = (
        extract_known_skills(
            resume_text
        )
    )


    skills.extend(
        extracted_skills
    )


    # --------------------------------------------------------
    # Normalize and deduplicate
    # --------------------------------------------------------

    return normalize_skills(
        skills
    )


# ============================================================
# CREATE JOB TEXT FOR SEMANTIC MATCHING
# ============================================================

def build_job_semantic_text(
    row
):
    """
    Create a richer representation of the job
    for semantic similarity.

    We include the job title, category,
    description and extracted requirements.
    """

    required_skills = row.get(
        "required_skills",
        []
    )


    preferred_skills = row.get(
        "preferred_skills",
        []
    )


    if not isinstance(
        required_skills,
        list
    ):

        required_skills = []


    if not isinstance(
        preferred_skills,
        list
    ):

        preferred_skills = []


    job_title = str(
        row.get(
            "Job_Title",
            ""
        )
    )


    category = str(
        row.get(
            "Category",
            ""
        )
    )


    description = str(
        row.get(
            "Job_Description",
            ""
        )
    )


    experience = str(
        row.get(
            "experience",
            ""
        )
    )


    education = str(
        row.get(
            "education",
            ""
        )
    )


    required_text = ", ".join(
        required_skills
    )


    preferred_text = ", ".join(
        preferred_skills
    )


    semantic_text = f"""
    Job Title:
    {job_title}

    Category:
    {category}

    Job Description:
    {description}

    Required Skills:
    {required_text}

    Preferred Skills:
    {preferred_text}

    Experience:
    {experience}

    Education:
    {education}
    """


    return semantic_text


# ============================================================
# BUILD RESUME SEMANTIC TEXT
# ============================================================

def build_resume_semantic_text(
    resume_text,
    resume_profile=None
):
    """
    Build a richer resume representation
    for semantic matching.
    """

    if not isinstance(
        resume_profile,
        dict
    ):

        return resume_text


    summary = resume_profile.get(
        "summary",
        ""
    )


    technical_skills = resume_profile.get(
        "technical_skills",
        []
    )


    soft_skills = resume_profile.get(
        "soft_skills",
        []
    )


    education = resume_profile.get(
        "education",
        []
    )


    experience = resume_profile.get(
        "experience",
        []
    )


    projects = resume_profile.get(
        "projects",
        []
    )


    education_text = json_safe_text(
        education
    )


    experience_text = json_safe_text(
        experience
    )


    projects_text = json_safe_text(
        projects
    )


    semantic_text = f"""
    Resume Summary:
    {summary}

    Technical Skills:
    {", ".join(technical_skills)}

    Soft Skills:
    {", ".join(soft_skills)}

    Education:
    {education_text}

    Experience:
    {experience_text}

    Projects:
    {projects_text}

    Original Resume:
    {resume_text}
    """


    return semantic_text


# ============================================================
# SAFE OBJECT TO TEXT
# ============================================================

def json_safe_text(
    value
):
    """
    Convert structured resume information
    into readable text.
    """

    if not value:

        return ""


    if isinstance(
        value,
        list
    ):

        parts = []


        for item in value:

            if isinstance(
                item,
                dict
            ):

                for key, item_value in item.items():

                    if isinstance(
                        item_value,
                        list
                    ):

                        item_value = ", ".join(
                            str(x)
                            for x in item_value
                        )


                    parts.append(
                        f"{key}: {item_value}"
                    )

            else:

                parts.append(
                    str(item)
                )


        return " | ".join(
            parts
        )


    if isinstance(
        value,
        dict
    ):

        return " | ".join(
            f"{key}: {item_value}"
            for key, item_value in value.items()
        )


    return str(
        value
    )


# ============================================================
# CALCULATE TF-IDF SIMILARITY
# ============================================================

def calculate_tfidf_similarity(
    resume_text,
    job_descriptions
):
    """
    Calculate TF-IDF cosine similarity
    between one resume and all jobs.
    """

    resume_vector = tfidf.transform(
        [resume_text]
    )


    job_vectors = tfidf.transform(
        job_descriptions
    )


    similarities = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]


    return similarities


# ============================================================
# CALCULATE SEMANTIC SIMILARITY
# ============================================================

def calculate_semantic_similarity(
    resume_semantic_text,
    job_semantic_texts
):
    """
    Calculate embedding-based semantic similarity.
    """

    model = get_embedding_model()


    resume_embedding = model.encode(
        [resume_semantic_text],
        normalize_embeddings=True
    )


    job_embeddings = model.encode(
        job_semantic_texts,
        normalize_embeddings=True
    )


    similarities = cosine_similarity(
        resume_embedding,
        job_embeddings
    )[0]


    return similarities


# ============================================================
# MATCH JOBS
# ============================================================

def match_jobs(
    resume_text,
    top_n=DEFAULT_TOP_N,
    resume_profile=None
):
    """
    Match a resume against all cached jobs.

    Returns a DataFrame containing:

        Similarity
        TFIDF_Similarity
        Semantic_Similarity
        Required_Skill_Coverage
        Preferred_Skill_Coverage
        Exact_Matches
        Related_Matches
        Missing_Skills
        Missing_Preferred_Skills

    Similarity is the final weighted ranking score.
    """

    if not resume_text:

        raise ValueError(
            "Resume text cannot be empty."
        )


    if not resume_text.strip():

        raise ValueError(
            "Resume text cannot be empty."
        )


    # ========================================================
    # LOAD JOBS
    # ========================================================

    jobs_df = get_jobs_dataframe()


    # ========================================================
    # PREPARE RESUME SKILLS
    # ========================================================

    resume_skills = prepare_resume_skills(
        resume_text,
        resume_profile
    )


    print(
        "\nResume skills used for matching:"
    )


    print(
        resume_skills
    )


    # ========================================================
    # TF-IDF SIMILARITY
    # ========================================================

    tfidf_scores = calculate_tfidf_similarity(

        resume_text,

        jobs_df[
            "Job_Description"
        ].astype(str)
    )


    jobs_df[
        "TFIDF_Similarity"
    ] = tfidf_scores


    # ========================================================
    # SEMANTIC SIMILARITY
    # ========================================================

    resume_semantic_text = (
        build_resume_semantic_text(
            resume_text,
            resume_profile
        )
    )


    job_semantic_texts = []


    for _, row in jobs_df.iterrows():

        job_semantic_texts.append(
            build_job_semantic_text(
                row
            )
        )


    semantic_scores = (
        calculate_semantic_similarity(

            resume_semantic_text,

            job_semantic_texts
        )
    )


    jobs_df[
        "Semantic_Similarity"
    ] = semantic_scores


    # ========================================================
    # SKILL MATCHING
    # ========================================================

    required_coverages = []

    preferred_coverages = []

    exact_matches_list = []

    related_matches_list = []

    missing_skills_list = []

    missing_preferred_list = []


    for _, row in jobs_df.iterrows():

        required_skills = row.get(
            "required_skills",
            []
        )


        preferred_skills = row.get(
            "preferred_skills",
            []
        )


        if not isinstance(
            required_skills,
            list
        ):

            required_skills = []


        if not isinstance(
            preferred_skills,
            list
        ):

            preferred_skills = []


        skill_result = analyze_job_skill_match(

            resume_skills,

            required_skills,

            preferred_skills
        )


        required_result = skill_result[
            "required"
        ]


        preferred_result = skill_result[
            "preferred"
        ]


        required_coverages.append(
            required_result[
                "coverage_score"
            ]
        )


        preferred_coverages.append(
            preferred_result[
                "coverage_score"
            ]
        )


        exact_matches_list.append(
            required_result[
                "exact_matches"
            ]
        )


        related_matches_list.append(
            required_result[
                "related_matches"
            ]
        )


        missing_skills_list.append(
            required_result[
                "missing_skills"
            ]
        )


        missing_preferred_list.append(
            preferred_result[
                "missing_skills"
            ]
        )


    jobs_df[
        "Required_Skill_Coverage"
    ] = required_coverages


    jobs_df[
        "Preferred_Skill_Coverage"
    ] = preferred_coverages


    jobs_df[
        "Exact_Matches"
    ] = exact_matches_list


    jobs_df[
        "Related_Matches"
    ] = related_matches_list


    jobs_df[
        "Missing_Skills"
    ] = missing_skills_list


    jobs_df[
        "Missing_Preferred_Skills"
    ] = missing_preferred_list


    # ========================================================
    # FINAL HYBRID SCORE
    # ========================================================

    jobs_df[
        "Similarity"
    ] = (

        jobs_df[
            "TFIDF_Similarity"
        ]
        * TFIDF_WEIGHT

        +

        jobs_df[
            "Semantic_Similarity"
        ]
        * SEMANTIC_WEIGHT

        +

        jobs_df[
            "Required_Skill_Coverage"
        ]
        * REQUIRED_SKILL_WEIGHT

        +

        jobs_df[
            "Preferred_Skill_Coverage"
        ]
        * PREFERRED_SKILL_WEIGHT
    )


    # ========================================================
    # SORT
    # ========================================================

    jobs_df = jobs_df.sort_values(

        by="Similarity",

        ascending=False
    )


    # ========================================================
    # TOP N
    # ========================================================

    top_jobs = jobs_df.head(
        top_n
    ).copy()


    # ========================================================
    # RANK
    # ========================================================

    top_jobs[
        "Rank"
    ] = range(
        1,
        len(top_jobs) + 1
    )


    # Put Rank first
    columns = [
        "Rank"
    ] + [
        column
        for column in top_jobs.columns
        if column != "Rank"
    ]


    top_jobs = top_jobs[
        columns
    ]


    return top_jobs


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    sample_resume = """

    Computer Science student with experience in
    Python, Java, SQL and Machine Learning.

    Built machine learning projects using
    Scikit-learn and Pandas.

    Experience with Power BI, Docker,
    Streamlit and data analysis.

    """

    print(
        "\n========================================"
    )

    print(
        "HYBRID JOB MATCHER TEST"
    )

    print(
        "========================================"
    )


    results = match_jobs(
        sample_resume,
        top_n=5
    )


    for _, row in results.iterrows():

        print(
            "\n----------------------------------------"
        )


        print(
            f"Rank: "
            f"{row['Rank']}"
        )


        print(
            f"Job: "
            f"{row['Job_Title']}"
        )


        print(
            f"Final Match Score: "
            f"{row['Similarity'] * 100:.2f}%"
        )


        print(
            f"TF-IDF: "
            f"{row['TFIDF_Similarity'] * 100:.2f}%"
        )


        print(
            f"Semantic: "
            f"{row['Semantic_Similarity'] * 100:.2f}%"
        )


        print(
            f"Required Skill Coverage: "
            f"{row['Required_Skill_Coverage'] * 100:.2f}%"
        )


        print(
            f"Preferred Skill Coverage: "
            f"{row['Preferred_Skill_Coverage'] * 100:.2f}%"
        )


        print(
            "Exact Matches:"
        )


        for skill in row[
            "Exact_Matches"
        ]:

            print(
                f"  ✓ {skill}"
            )


        print(
            "Related Matches:"
        )


        for item in row[
            "Related_Matches"
        ]:

            print(
                f"  ~ "
                f"{item['required_skill']} "
                f"<- "
                f"{', '.join(item['resume_skills'])}"
            )


        print(
            "Missing Skills:"
        )


        for skill in row[
            "Missing_Skills"
        ]:

            print(
                f"  ✗ {skill}"
            )
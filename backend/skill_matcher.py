from backend.skill_normalizer import (
    normalize_skill,
    normalize_skills
)


# ============================================================
# RELATED SKILL RELATIONSHIPS
# ============================================================
#
# IMPORTANT:
#
# Related does NOT mean equivalent.
#
# We only use a related relationship when the resume skill
# provides meaningful evidence for the job skill.
#
# Example:
#
#     Job requires:
#         Machine Learning
#
#     Resume:
#         Scikit-learn
#
#     Result:
#         RELATED
#
# But:
#
#     Job requires:
#         Data Structures & Algorithms
#
#     Resume:
#         Python
#
#     Result:
#         NOT RELATED
#
# We deliberately keep this conservative.
# ============================================================

RELATED_SKILLS = {

    # --------------------------------------------------------
    # Machine Learning
    # --------------------------------------------------------

    "Machine Learning": {
        "Scikit-learn",
        "TensorFlow",
        "PyTorch",
        "Keras",
        "XGBoost"
    },


    # --------------------------------------------------------
    # Deep Learning
    # --------------------------------------------------------

    "Deep Learning": {
        "TensorFlow",
        "PyTorch",
        "Keras"
    },


    # --------------------------------------------------------
    # Artificial Intelligence
    # --------------------------------------------------------

    "Artificial Intelligence": {
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Large Language Models",
        "Generative AI"
    },


    # --------------------------------------------------------
    # Generative AI
    # --------------------------------------------------------

    "Generative AI": {
        "Large Language Models",
        "Retrieval-Augmented Generation",
        "Prompt Engineering"
    },


    # --------------------------------------------------------
    # Large Language Models
    # --------------------------------------------------------

    "Large Language Models": {
        "Generative AI",
        "Retrieval-Augmented Generation",
        "Prompt Engineering",
        "Embeddings"
    },


    # --------------------------------------------------------
    # Natural Language Processing
    # --------------------------------------------------------

    "Natural Language Processing": {
        "Large Language Models",
        "Sentence Transformers",
        "Embeddings",
        "Retrieval-Augmented Generation"
    },


    # --------------------------------------------------------
    # Data Analysis
    # --------------------------------------------------------

    "Data Analysis": {
        "Pandas",
        "NumPy",
        "SQL",
        "Excel",
        "Power BI",
        "Tableau",
        "Statistics"
    },


    # --------------------------------------------------------
    # Data Visualization
    # --------------------------------------------------------

    "Data Visualization": {
        "Power BI",
        "Tableau",
        "Matplotlib",
        "Seaborn"
    },


    # --------------------------------------------------------
    # SQL / Databases
    # --------------------------------------------------------

    "SQL": {
        "MySQL",
        "PostgreSQL",
        "Microsoft SQL Server",
        "Oracle",
        "SQLite"
    },


    "Database Management Systems": {
        "SQL",
        "MySQL",
        "PostgreSQL",
        "Microsoft SQL Server",
        "MongoDB",
        "Oracle"
    },


    # --------------------------------------------------------
    # Backend Development
    # --------------------------------------------------------
    #
    # We are deliberately NOT saying:
    #
    # Python → Backend Development
    #
    # Instead, actual backend technologies provide evidence.
    # --------------------------------------------------------

    "Backend Development": {
        "Django",
        "Flask",
        "FastAPI",
        "Node.js"
    },


    # --------------------------------------------------------
    # REST API
    # --------------------------------------------------------

    "REST API": {
        "FastAPI",
        "Flask",
        "Django",
        "Node.js"
    },


    # --------------------------------------------------------
    # Cloud Deployment
    # --------------------------------------------------------

    "Cloud Deployment": {
        "AWS",
        "Microsoft Azure",
        "Google Cloud",
        "Docker",
        "Kubernetes"
    },


    # --------------------------------------------------------
    # DevOps
    # --------------------------------------------------------

    "DevOps": {
        "Docker",
        "Kubernetes",
        "Jenkins",
        "AWS",
        "Microsoft Azure",
        "Google Cloud"
    },


    # --------------------------------------------------------
    # Vector Search
    # --------------------------------------------------------

    "Vector Search": {
        "FAISS",
        "ChromaDB",
        "Embeddings",
        "Sentence Transformers"
    },


    # --------------------------------------------------------
    # Retrieval-Augmented Generation
    # --------------------------------------------------------

    "Retrieval-Augmented Generation": {
        "Embeddings",
        "Vector Search",
        "Sentence Transformers",
        "Large Language Models"
    },


    # --------------------------------------------------------
    # Data Structures & Algorithms
    # --------------------------------------------------------
    #
    # Intentionally EMPTY.
    #
    # Knowing a programming language is not enough evidence
    # to claim DSA knowledge.
    # --------------------------------------------------------

    "Data Structures & Algorithms": set()
}


# ============================================================
# ADDITIONAL NORMALIZATION FOR JOB REQUIREMENTS
# ============================================================

MATCHING_ALIASES = {

    # --------------------------------------------------------
    # DSA
    # --------------------------------------------------------

    "data structures":
        "Data Structures & Algorithms",

    "algorithms":
        "Data Structures & Algorithms",

    "data structures and algorithms":
        "Data Structures & Algorithms",

    "data structures & algorithms":
        "Data Structures & Algorithms",


    # --------------------------------------------------------
    # SQL
    # --------------------------------------------------------

    "sql databases":
        "SQL",

    "sql database":
        "SQL",


    # --------------------------------------------------------
    # Backend
    # --------------------------------------------------------

    "backend development":
        "Backend Development",

    "backend":
        "Backend Development",


    # --------------------------------------------------------
    # Cloud
    # --------------------------------------------------------

    "cloud deployment":
        "Cloud Deployment",


    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    "model training":
        "Machine Learning",

    "training machine learning models":
        "Machine Learning",

    "training ml models":
        "Machine Learning",

    "ml model training":
        "Machine Learning",


    # --------------------------------------------------------
    # Model Deployment
    # --------------------------------------------------------

    "model deployment":
        "Model Deployment",

    "deploying ml models":
        "Model Deployment",

    "deploying ml models in production":
        "Model Deployment",

    "ml model deployment":
        "Model Deployment",

    "production model deployment":
        "Model Deployment",


    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    "reporting":
        "Reporting",

    "reporting dashboards":
        "Reporting",

    "dashboard reporting":
        "Reporting",


    # --------------------------------------------------------
    # Business Intelligence
    # --------------------------------------------------------

    "business intelligence":
        "Business Intelligence",

    "business intelligence insights":
        "Business Intelligence",


    # --------------------------------------------------------
    # Data Analysis
    # --------------------------------------------------------

    "data analysis":
        "Data Analysis",

    "data analytics":
        "Data Analysis",


    # --------------------------------------------------------
    # Data Cleaning
    # --------------------------------------------------------

    "data cleaning":
        "Data Cleaning",


    # --------------------------------------------------------
    # Statistical Analysis
    # --------------------------------------------------------

    "statistical analysis":
        "Statistical Analysis"
}


# ============================================================
# NORMALIZE ONE SKILL FOR MATCHING
# ============================================================

def normalize_for_matching(
    skill
):
    """
    Normalize a single skill using:

    1. Matching-specific aliases
    2. General skill normalizer
    """

    if not skill:

        return None


    raw_skill = str(
        skill
    ).strip()


    if not raw_skill:

        return None


    lookup_key = (
        raw_skill.lower()
        .strip()
    )


    # --------------------------------------------------------
    # Matching-specific aliases
    # --------------------------------------------------------

    if lookup_key in MATCHING_ALIASES:

        return MATCHING_ALIASES[
            lookup_key
        ]


    # --------------------------------------------------------
    # General aliases
    # --------------------------------------------------------

    return normalize_skill(
        raw_skill
    )


# ============================================================
# NORMALIZE MULTIPLE SKILLS
# ============================================================

def normalize_matching_skills(
    skills
):
    """
    Normalize and deduplicate a list of skills.
    """

    if not skills:

        return []


    normalized = []

    seen = set()


    for skill in skills:

        canonical = (
            normalize_for_matching(
                skill
            )
        )


        if not canonical:

            continue


        key = canonical.lower()


        if key in seen:

            continue


        seen.add(
            key
        )


        normalized.append(
            canonical
        )


    return normalized


# ============================================================
# EXACT MATCHES
# ============================================================

def find_exact_matches(
    resume_skills,
    job_skills
):
    """
    Find skills explicitly present in both
    the resume and job requirements.
    """

    resume_normalized = set(
        normalize_matching_skills(
            resume_skills
        )
    )


    job_normalized = set(
        normalize_matching_skills(
            job_skills
        )
    )


    return sorted(
        resume_normalized.intersection(
            job_normalized
        )
    )


# ============================================================
# RELATED MATCHES
# ============================================================

def find_related_matches(
    resume_skills,
    job_skills,
    exact_matches=None
):
    """
    Find meaningful related-skill evidence.
    """

    resume_normalized = set(
        normalize_matching_skills(
            resume_skills
        )
    )


    job_normalized = set(
        normalize_matching_skills(
            job_skills
        )
    )


    if exact_matches is None:

        exact_matches = find_exact_matches(
            resume_skills,
            job_skills
        )


    exact_set = set(
        exact_matches
    )


    related_matches = []


    for required_skill in sorted(
        job_normalized
    ):

        # ----------------------------------------------------
        # Exact matches should never also be related.
        # ----------------------------------------------------

        if required_skill in exact_set:

            continue


        related_skills = RELATED_SKILLS.get(
            required_skill,
            set()
        )


        candidate_related_skills = (
            sorted(
                resume_normalized.intersection(
                    related_skills
                )
            )
        )


        if candidate_related_skills:

            related_matches.append(

                {
                    "required_skill":
                        required_skill,

                    "resume_skills":
                        candidate_related_skills
                }
            )


    return related_matches


# ============================================================
# MISSING SKILLS
# ============================================================

def find_missing_skills(
    resume_skills,
    job_skills,
    exact_matches=None,
    related_matches=None
):
    """
    Find job skills with no exact or related evidence.
    """

    job_normalized = set(
        normalize_matching_skills(
            job_skills
        )
    )


    if exact_matches is None:

        exact_matches = find_exact_matches(
            resume_skills,
            job_skills
        )


    if related_matches is None:

        related_matches = find_related_matches(
            resume_skills,
            job_skills,
            exact_matches
        )


    covered_skills = set(
        exact_matches
    )


    for item in related_matches:

        covered_skills.add(
            item[
                "required_skill"
            ]
        )


    return sorted(
        job_normalized.difference(
            covered_skills
        )
    )


# ============================================================
# SKILL COVERAGE
# ============================================================

def calculate_skill_coverage(
    resume_skills,
    job_skills
):
    """
    Calculate weighted skill coverage.

    Exact:
        1.0

    Related:
        0.5

    Missing:
        0.0
    """

    normalized_job_skills = (
        normalize_matching_skills(
            job_skills
        )
    )


    if not normalized_job_skills:

        return {

            "coverage_score": 0.0,

            "exact_matches": [],

            "related_matches": [],

            "missing_skills": [],

            "total_job_skills": 0
        }


    exact_matches = find_exact_matches(
        resume_skills,
        normalized_job_skills
    )


    related_matches = find_related_matches(
        resume_skills,
        normalized_job_skills,
        exact_matches
    )


    missing_skills = find_missing_skills(
        resume_skills,
        normalized_job_skills,
        exact_matches,
        related_matches
    )


    exact_count = len(
        exact_matches
    )


    related_count = len(
        related_matches
    )


    total_skills = len(
        normalized_job_skills
    )


    weighted_coverage = (
        exact_count
        +
        (0.5 * related_count)
    )


    coverage_score = (
        weighted_coverage
        /
        total_skills
    )


    return {

        "coverage_score":
            coverage_score,

        "exact_matches":
            exact_matches,

        "related_matches":
            related_matches,

        "missing_skills":
            missing_skills,

        "total_job_skills":
            total_skills
    }


# ============================================================
# COMPLETE JOB SKILL ANALYSIS
# ============================================================

def analyze_job_skill_match(
    resume_skills,
    required_skills,
    preferred_skills
):
    """
    Analyze required and preferred skills separately.
    """

    required_result = calculate_skill_coverage(

        resume_skills,

        required_skills
    )


    preferred_result = calculate_skill_coverage(

        resume_skills,

        preferred_skills
    )


    return {

        "required":
            required_result,

        "preferred":
            preferred_result
    }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )

    print(
        "SKILL MATCHER TEST"
    )

    print(
        "========================================"
    )


    resume_skills = [

        "Python",

        "SQL",

        "Scikit-learn",

        "Pandas",

        "Power BI",

        "Docker"
    ]


    required_skills = [

        "Python",

        "Machine Learning",

        "SQL",

        "Excel",

        "Data Analysis",

        "Cloud Deployment",

        "Data Structures & Algorithms"
    ]


    preferred_skills = [

        "Tableau",

        "Statistics"
    ]


    result = analyze_job_skill_match(

        resume_skills,

        required_skills,

        preferred_skills
    )


    required = result[
        "required"
    ]


    print(
        "\nREQUIRED SKILLS"
    )


    print(
        f"Coverage: "
        f"{required['coverage_score'] * 100:.2f}%"
    )


    print(
        "\nExact matches:"
    )


    for skill in required[
        "exact_matches"
    ]:

        print(
            f"✓ {skill}"
        )


    print(
        "\nRelated matches:"
    )


    for item in required[
        "related_matches"
    ]:

        print(
            f"~ {item['required_skill']} "
            f"<- "
            f"{', '.join(item['resume_skills'])}"
        )


    print(
        "\nMissing skills:"
    )


    for skill in required[
        "missing_skills"
    ]:

        print(
            f"✗ {skill}"
        )


    preferred = result[
        "preferred"
    ]


    print(
        "\nPREFERRED SKILLS"
    )


    print(
        f"Coverage: "
        f"{preferred['coverage_score'] * 100:.2f}%"
    )


    print(
        "\nExact matches:"
    )


    for skill in preferred[
        "exact_matches"
    ]:

        print(
            f"✓ {skill}"
        )


    print(
        "\nRelated matches:"
    )


    for item in preferred[
        "related_matches"
    ]:

        print(
            f"~ {item['required_skill']} "
            f"<- "
            f"{', '.join(item['resume_skills'])}"
        )


    print(
        "\nMissing skills:"
    )


    for skill in preferred[
        "missing_skills"
    ]:

        print(
            f"✗ {skill}"
        )
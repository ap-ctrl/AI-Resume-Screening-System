import re


# ============================================================
# CANONICAL SKILL ALIASES
# ============================================================

SKILL_ALIASES = {

    # --------------------------------------------------------
    # Programming Languages
    # --------------------------------------------------------

    "python": "Python",

    "java": "Java",

    "c++": "C++",

    "cpp": "C++",

    "c plus plus": "C++",

    "c#": "C#",

    "c sharp": "C#",

    "javascript": "JavaScript",

    "java script": "JavaScript",

    "js": "JavaScript",

    "typescript": "TypeScript",

    "ts": "TypeScript",

    "kotlin": "Kotlin",

    "swift": "Swift",

    "r programming": "R",

    "r language": "R",


    # --------------------------------------------------------
    # Databases
    # --------------------------------------------------------

    "sql": "SQL",

    "mysql": "MySQL",

    "mssql": "Microsoft SQL Server",

    "sql server": "Microsoft SQL Server",

    "microsoft sql server": "Microsoft SQL Server",

    "postgres": "PostgreSQL",

    "postgresql": "PostgreSQL",

    "mongodb": "MongoDB",

    "mongo db": "MongoDB",

    "oracle": "Oracle",

    "redis": "Redis",

    "sqlite": "SQLite",


    # --------------------------------------------------------
    # Machine Learning / AI
    # --------------------------------------------------------

    "ml": "Machine Learning",

    "machine learning": "Machine Learning",

    "machine-learning": "Machine Learning",

    "ai": "Artificial Intelligence",

    "artificial intelligence": "Artificial Intelligence",

    "deep learning": "Deep Learning",

    "dl": "Deep Learning",

    "nlp": "Natural Language Processing",

    "natural language processing":
        "Natural Language Processing",

    "computer vision": "Computer Vision",

    "cv": "Computer Vision",

    "llm": "Large Language Models",

    "llms": "Large Language Models",

    "large language model":
        "Large Language Models",

    "large language models":
        "Large Language Models",

    "generative ai": "Generative AI",

    "genai": "Generative AI",

    "rag": "Retrieval-Augmented Generation",

    "retrieval augmented generation":
        "Retrieval-Augmented Generation",

    "retrieval-augmented generation":
        "Retrieval-Augmented Generation",


    # --------------------------------------------------------
    # Machine Learning Libraries
    # --------------------------------------------------------

    "sklearn": "Scikit-learn",

    "scikit learn": "Scikit-learn",

    "scikit-learn": "Scikit-learn",

    "scikit_learn": "Scikit-learn",

    "tensorflow": "TensorFlow",

    "tf": "TensorFlow",

    "pytorch": "PyTorch",

    "torch": "PyTorch",

    "keras": "Keras",

    "pandas": "Pandas",

    "numpy": "NumPy",

    "matplotlib": "Matplotlib",

    "seaborn": "Seaborn",

    "opencv": "OpenCV",


    # --------------------------------------------------------
    # NLP / Embeddings
    # --------------------------------------------------------

    "sentence transformers":
        "Sentence Transformers",

    "sentence-transformers":
        "Sentence Transformers",

    "sentence transformer":
        "Sentence Transformers",

    "embeddings": "Embeddings",

    "embedding": "Embeddings",

    "vector search": "Vector Search",

    "vector database": "Vector Database",

    "vector databases": "Vector Database",

    "chromadb": "ChromaDB",

    "chroma db": "ChromaDB",

    "faiss": "FAISS",


    # --------------------------------------------------------
    # Cloud / DevOps
    # --------------------------------------------------------

    "aws": "AWS",

    "amazon web services": "AWS",

    "azure": "Microsoft Azure",

    "microsoft azure": "Microsoft Azure",

    "gcp": "Google Cloud",

    "google cloud": "Google Cloud",

    "docker": "Docker",

    "docker compose": "Docker Compose",

    "kubernetes": "Kubernetes",

    "k8s": "Kubernetes",

    "jenkins": "Jenkins",


    # --------------------------------------------------------
    # Web / Backend
    # --------------------------------------------------------

    "html": "HTML",

    "html5": "HTML",

    "css": "CSS",

    "css3": "CSS",

    "react": "React",

    "reactjs": "React",

    "node": "Node.js",

    "nodejs": "Node.js",

    "node.js": "Node.js",

    "express": "Express.js",

    "expressjs": "Express.js",

    "fastapi": "FastAPI",

    "flask": "Flask",

    "django": "Django",

    "rest api": "REST API",

    "rest apis": "REST API",

    "restful api": "REST API",


    # --------------------------------------------------------
    # Tools
    # --------------------------------------------------------

    "git": "Git",

    "github": "GitHub",

    "gitlab": "GitLab",

    "vs code": "VS Code",

    "visual studio code": "VS Code",

    "google colab": "Google Colab",

    "colab": "Google Colab",

    "jupyter": "Jupyter",

    "jupyter notebook": "Jupyter Notebook",

    "streamlit": "Streamlit",

    "power bi": "Power BI",

    "powerbi": "Power BI",

    "tableau": "Tableau",

    "excel": "Excel",

    "microsoft excel": "Excel",

    "openpyxl": "OpenPyXL",


    # --------------------------------------------------------
    # Data / Analytics
    # --------------------------------------------------------

    "data analysis": "Data Analysis",

    "data analytics": "Data Analytics",

    "data visualization": "Data Visualization",

    "data cleaning": "Data Cleaning",

    "data preprocessing": "Data Preprocessing",

    "data mining": "Data Mining",

    "data warehousing": "Data Warehousing",

    "statistics": "Statistics",

    "statistical analysis": "Statistical Analysis",

    "business intelligence": "Business Intelligence",

    "bi": "Business Intelligence",


    # --------------------------------------------------------
    # Core CS
    # --------------------------------------------------------

    "dsa": "Data Structures & Algorithms",

    "data structures and algorithms":
        "Data Structures & Algorithms",

    "data structures & algorithms":
        "Data Structures & Algorithms",

    "object oriented programming":
        "Object-Oriented Programming",

    "object-oriented programming":
        "Object-Oriented Programming",

    "oop": "Object-Oriented Programming",

    "operating system": "Operating Systems",

    "operating systems": "Operating Systems",

    "dbms": "Database Management Systems",

    "database management system":
        "Database Management Systems",

    "database management systems":
        "Database Management Systems",

    "computer networks": "Computer Networks",

    "system design": "System Design",


    # --------------------------------------------------------
    # LLM / Agent Frameworks
    # --------------------------------------------------------

    "langgraph": "LangGraph",

    "langchain": "LangChain",

    "ollama": "Ollama",

    "groq": "Groq",

    "groq api": "Groq API",

    "prompt engineering": "Prompt Engineering",

    "knowledge graph": "Knowledge Graphs",

    "knowledge graphs": "Knowledge Graphs"
}


# ============================================================
# NORMALIZE A SINGLE SKILL
# ============================================================

def normalize_skill(skill):
    """
    Convert a skill name into its canonical form.

    Example:

        ML
        ↓
        Machine Learning

        sklearn
        ↓
        Scikit-learn
    """

    if not skill:

        return None


    # Convert to string
    skill = str(skill)


    # Remove unnecessary whitespace
    skill = skill.strip()


    if not skill:

        return None


    # Lowercase for alias lookup
    lookup_key = skill.lower()


    # Normalize repeated whitespace
    lookup_key = re.sub(
        r"\s+",
        " ",
        lookup_key
    )


    # Check alias dictionary
    if lookup_key in SKILL_ALIASES:

        return SKILL_ALIASES[
            lookup_key
        ]


    # If no alias exists, preserve the original
    # while cleaning its formatting.

    return skill.strip()


# ============================================================
# NORMALIZE MULTIPLE SKILLS
# ============================================================

def normalize_skills(skills):
    """
    Normalize a list of skills and remove duplicates.

    Parameters:
        skills: List of raw skill names.

    Returns:
        List of canonical skill names.
    """

    if not skills:

        return []


    normalized = []

    seen = set()


    for skill in skills:

        canonical_skill = normalize_skill(
            skill
        )


        if not canonical_skill:

            continue


        # Case-insensitive duplicate detection
        duplicate_key = (
            canonical_skill.lower()
        )


        if duplicate_key in seen:

            continue


        seen.add(
            duplicate_key
        )

        normalized.append(
            canonical_skill
        )


    return normalized


# ============================================================
# NORMALIZE RESUME PROFILE
# ============================================================

def normalize_resume_profile(
    resume_profile
):
    """
    Normalize the skills contained inside
    a structured resume profile.

    This function does not modify the original
    profile object.

    Returns:
        A new normalized profile.
    """

    if not isinstance(
        resume_profile,
        dict
    ):

        return resume_profile


    normalized_profile = (
        resume_profile.copy()
    )


    # --------------------------------------------------------
    # Technical skills
    # --------------------------------------------------------

    normalized_profile[
        "technical_skills"
    ] = normalize_skills(
        resume_profile.get(
            "technical_skills",
            []
        )
    )


    # --------------------------------------------------------
    # Soft skills
    # --------------------------------------------------------

    normalized_profile[
        "soft_skills"
    ] = normalize_skills(
        resume_profile.get(
            "soft_skills",
            []
        )
    )


    # --------------------------------------------------------
    # Project technologies
    # --------------------------------------------------------

    projects = resume_profile.get(
        "projects",
        []
    )


    normalized_projects = []


    for project in projects:

        if not isinstance(
            project,
            dict
        ):

            normalized_projects.append(
                project
            )

            continue


        normalized_project = (
            project.copy()
        )


        normalized_project[
            "technologies"
        ] = normalize_skills(
            project.get(
                "technologies",
                []
            )
        )


        normalized_projects.append(
            normalized_project
        )


    normalized_profile[
        "projects"
    ] = normalized_projects


    return normalized_profile


# ============================================================
# FIND KNOWN SKILLS IN TEXT
# ============================================================

def extract_known_skills(
    text
):
    """
    Find known skills from free-form text.

    This is intentionally deterministic.
    It does not use an LLM.

    Returns:
        List of canonical skill names.
    """

    if not text:

        return []


    text_lower = text.lower()


    found_skills = []


    # Sort aliases by length so that
    # multi-word skills are checked first.

    aliases = sorted(
        SKILL_ALIASES.items(),
        key=lambda item: len(
            item[0]
        ),
        reverse=True
    )


    for alias, canonical_skill in aliases:

        # ----------------------------------------------------
        # Build a safe regex.
        #
        # Word boundaries help avoid matching:
        #
        # "r" inside "program"
        #
        # or:
        #
        # "c" inside "cloud"
        # ----------------------------------------------------

        escaped_alias = re.escape(
            alias
        )


        pattern = (
            r"(?<![a-zA-Z0-9])"
            + escaped_alias
            + r"(?![a-zA-Z0-9])"
        )


        if re.search(
            pattern,
            text_lower
        ):

            found_skills.append(
                canonical_skill
            )


    return normalize_skills(
        found_skills
    )


# ============================================================
# COMPARE SKILL SETS
# ============================================================

def compare_skills(
    resume_skills,
    job_skills
):
    """
    Compare two skill lists.

    Returns:
        matched_skills
        missing_skills
        extra_resume_skills
    """

    resume_normalized = set(
        normalize_skills(
            resume_skills
        )
    )


    job_normalized = set(
        normalize_skills(
            job_skills
        )
    )


    matched_skills = sorted(
        resume_normalized.intersection(
            job_normalized
        )
    )


    missing_skills = sorted(
        job_normalized.difference(
            resume_normalized
        )
    )


    extra_resume_skills = sorted(
        resume_normalized.difference(
            job_normalized
        )
    )


    return (
        matched_skills,
        missing_skills,
        extra_resume_skills
    )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )

    print(
        "SKILL NORMALIZER TEST"
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # Test 1: Individual skills
    # --------------------------------------------------------

    test_skills = [
        "Python",
        "ML",
        "machine-learning",
        "sklearn",
        "scikit learn",
        "SQL",
        "powerbi",
        "js",
        "docker",
        "rag"
    ]


    print(
        "\nOriginal skills:"
    )

    print(
        test_skills
    )


    normalized = normalize_skills(
        test_skills
    )


    print(
        "\nNormalized skills:"
    )

    for skill in normalized:

        print(
            f"✓ {skill}"
        )


    # --------------------------------------------------------
    # Test 2: Extract skills from text
    # --------------------------------------------------------

    sample_text = """
    I have experience with Python,
    machine learning, sklearn, SQL,
    Power BI, Docker, RAG and JavaScript.
    """


    extracted = extract_known_skills(
        sample_text
    )


    print(
        "\nSkills extracted from text:"
    )


    for skill in extracted:

        print(
            f"✓ {skill}"
        )


    # --------------------------------------------------------
    # Test 3: Compare skills
    # --------------------------------------------------------

    resume_skills = [
        "Python",
        "ML",
        "sklearn",
        "SQL",
        "Power BI"
    ]


    job_skills = [
        "Python",
        "Machine Learning",
        "Scikit-learn",
        "SQL",
        "Excel",
        "Tableau"
    ]


    matched, missing, extra = (
        compare_skills(
            resume_skills,
            job_skills
        )
    )


    print(
        "\nMatched skills:"
    )

    for skill in matched:

        print(
            f"✓ {skill}"
        )


    print(
        "\nMissing skills:"
    )

    for skill in missing:

        print(
            f"✗ {skill}"
        )


    print(
        "\nAdditional resume skills:"
    )

    for skill in extra:

        print(
            f"+ {skill}"
        )
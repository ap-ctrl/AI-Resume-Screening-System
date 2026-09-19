import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

_model = None


def get_embedding_model():
    """
    Load the Sentence Transformer model.

    The model is loaded only once and reused for
    subsequent requests.
    """

    global _model

    if _model is None:

        print(
            f"Loading embedding model: {MODEL_NAME}"
        )

        _model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "Embedding model loaded successfully."
        )

    return _model


# ============================================================
# GENERATE EMBEDDING
# ============================================================

def generate_embedding(text):
    """
    Convert text into a semantic embedding.

    Parameters:
        text: Text to encode.

    Returns:
        Numpy array containing the embedding.
    """

    if not text or not text.strip():

        raise ValueError(
            "Text cannot be empty when generating an embedding."
        )

    model = get_embedding_model()

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding


# ============================================================
# CALCULATE SEMANTIC SIMILARITY
# ============================================================

def calculate_semantic_similarity(
    resume_text,
    job_text
):
    """
    Calculate semantic similarity between a resume
    and a job description.

    Returns:
        Similarity value between approximately 0 and 1.
    """

    resume_embedding = generate_embedding(
        resume_text
    )

    job_embedding = generate_embedding(
        job_text
    )

    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    # Convert numpy value to normal Python float.
    similarity = float(similarity)

    # Keep score within the expected range.
    similarity = max(
        0.0,
        min(
            1.0,
            similarity
        )
    )

    return similarity


# ============================================================
# CALCULATE SIMILARITY AGAINST MULTIPLE JOBS
# ============================================================

def calculate_job_similarities(
    resume_text,
    job_descriptions
):
    """
    Calculate semantic similarity between one resume
    and multiple job descriptions.

    Parameters:
        resume_text: Resume text.
        job_descriptions: List or Series of job descriptions.

    Returns:
        Numpy array containing one similarity score
        for each job.
    """

    if not resume_text or not resume_text.strip():

        raise ValueError(
            "Resume text cannot be empty."
        )

    model = get_embedding_model()

    # --------------------------------------------------------
    # Generate resume embedding
    # --------------------------------------------------------

    resume_embedding = model.encode(
        resume_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Generate embeddings for all jobs
    # --------------------------------------------------------

    job_texts = [
        str(job)
        for job in job_descriptions
    ]

    job_embeddings = model.encode(
        job_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        [resume_embedding],
        job_embeddings
    )[0]

    # --------------------------------------------------------
    # Convert and clip values
    # --------------------------------------------------------

    similarities = np.asarray(
        similarities,
        dtype=float
    )

    similarities = np.clip(
        similarities,
        0.0,
        1.0
    )

    return similarities


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    resume = """
    Computer Science student with experience in Python,
    machine learning, SQL, data analysis and building
    machine learning projects.
    """

    job = """
    Looking for a candidate experienced in developing
    machine learning solutions using Python and working
    with data.
    """

    score = calculate_semantic_similarity(
        resume,
        job
    )

    print(
        f"Semantic Similarity: {score:.4f}"
    )

    print(
        f"Semantic Similarity Percentage: "
        f"{score * 100:.2f}%"
    )
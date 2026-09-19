"""
Automatically generate a baseline ground-truth file for job matching.

IMPORTANT:
This is a CATEGORY-BASED BASELINE.

A job is considered relevant when:
    resume category == job category

This is NOT human-annotated ground truth.
It is an initial evaluation baseline that lets us measure the
matching system without manually labeling every resume.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESUME_DATA_PATH = (
    PROJECT_ROOT / "data" / "resume_dataset.csv"
)

JOBS_DATA_PATH = (
    PROJECT_ROOT / "data" / "jobs.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "evaluation"

OUTPUT_PATH = (
    OUTPUT_DIR / "job_matching_ground_truth.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load resume and job datasets."""

    if not RESUME_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Resume dataset not found:\n{RESUME_DATA_PATH}"
        )

    if not JOBS_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Jobs dataset not found:\n{JOBS_DATA_PATH}"
        )

    resumes = pd.read_csv(
        RESUME_DATA_PATH
    )

    jobs = pd.read_csv(
        JOBS_DATA_PATH
    )

    return resumes, jobs


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(resumes, jobs):
    """Check required columns."""

    required_resume_columns = {
        "ID",
        "Resume_str",
        "Category",
    }

    required_job_columns = {
        "Job_Title",
        "Category",
        "Job_Description",
    }

    missing_resume_columns = (
        required_resume_columns
        - set(resumes.columns)
    )

    missing_job_columns = (
        required_job_columns
        - set(jobs.columns)
    )

    if missing_resume_columns:
        raise ValueError(
            "Resume dataset is missing columns: "
            + ", ".join(
                sorted(missing_resume_columns)
            )
        )

    if missing_job_columns:
        raise ValueError(
            "Jobs dataset is missing columns: "
            + ", ".join(
                sorted(missing_job_columns)
            )
        )


# ============================================================
# CREATE BASELINE GROUND TRUTH
# ============================================================

def create_ground_truth(resumes, jobs):
    """
    Create category-based relevance labels.

    For every resume:
        relevant jobs = jobs with the same category.

    Example:

        Resume category:
        ACCOUNTANT

        Jobs:
        Accountant -> relevant

        Banking Officer -> not relevant
    """

    # Clean category values.
    resumes = resumes.copy()
    jobs = jobs.copy()

    resumes["Category"] = (
        resumes["Category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    jobs["Category"] = (
        jobs["Category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Create a dictionary:
    #
    # category -> list of jobs
    #
    jobs_by_category = {}

    for category in jobs["Category"].unique():

        category_jobs = jobs[
            jobs["Category"] == category
        ]

        job_titles = (
            category_jobs["Job_Title"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        jobs_by_category[category] = job_titles

    rows = []

    for _, resume in resumes.iterrows():

        resume_category = resume["Category"]

        relevant_jobs = jobs_by_category.get(
            resume_category,
            []
        )

        # Only create an evaluation row when
        # at least one job exists for that category.
        if not relevant_jobs:
            continue

        rows.append(
            {
                "resume_id": resume["ID"],
                "category": resume_category,
                "relevant_jobs": "|".join(
                    relevant_jobs
                ),
                "review_status": "APPROVED",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUTOMATIC GROUND-TRUTH GENERATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading datasets...")

    resumes, jobs = load_data()

    print(
        f"Resumes loaded: {len(resumes)}"
    )

    print(
        f"Jobs loaded: {len(jobs)}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_data(
        resumes,
        jobs
    )

    # --------------------------------------------------------
    # Create ground truth
    # --------------------------------------------------------

    print(
        "\nCreating category-based relevance labels..."
    )

    ground_truth = create_ground_truth(
        resumes,
        jobs
    )

    if ground_truth.empty:

        print(
            "\nNo matching resume/job categories found."
        )

        return

    # --------------------------------------------------------
    # Create evaluation directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    ground_truth.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GROUND TRUTH CREATED")
    print("=" * 70)

    print(
        f"\nEvaluation resumes: "
        f"{len(ground_truth)}"
    )

    print(
        f"Categories covered: "
        f"{ground_truth['category'].nunique()}"
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("\nExample rows:")

    print(
        ground_truth.head(10).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)

    print(
        "IMPORTANT:"
    )

    print(
        "This is a category-based baseline."
    )

    print(
        "It should not be described as human-annotated "
        "ground truth."
    )

    print(
        "Later, we can manually validate a smaller sample."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()







# """
# Generate a ground-truth template for evaluating the job matching system.

# This script:
# 1. Reads the resume dataset.
# 2. Reads the jobs dataset.
# 3. Finds job categories that exist in both datasets.
# 4. Selects a small number of resumes from each matching category.
# 5. Creates a CSV where we can manually mark which jobs are actually relevant.

# IMPORTANT:
# This script does NOT decide that category matching means a job is relevant.
# It only creates candidates for manual review.
# """

# from pathlib import Path

# import pandas as pd


# # ============================================================
# # PROJECT PATHS
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RESUME_DATA_PATH = (
#     PROJECT_ROOT / "data" / "resume_dataset.csv"
# )

# JOBS_DATA_PATH = (
#     PROJECT_ROOT / "data" / "jobs.csv"
# )

# OUTPUT_DIR = PROJECT_ROOT / "evaluation"

# OUTPUT_PATH = (
#     OUTPUT_DIR / "job_matching_ground_truth.csv"
# )


# # ============================================================
# # SETTINGS
# # ============================================================

# # Number of resumes we want from each category.
# # Starting small makes manual review easier.
# RESUMES_PER_CATEGORY = 1


# # ============================================================
# # LOAD DATA
# # ============================================================

# def load_data():
#     """Load the resume and job datasets."""

#     if not RESUME_DATA_PATH.exists():
#         raise FileNotFoundError(
#             f"Resume dataset not found:\n{RESUME_DATA_PATH}"
#         )

#     if not JOBS_DATA_PATH.exists():
#         raise FileNotFoundError(
#             f"Jobs dataset not found:\n{JOBS_DATA_PATH}"
#         )

#     resumes = pd.read_csv(RESUME_DATA_PATH)
#     jobs = pd.read_csv(JOBS_DATA_PATH)

#     return resumes, jobs


# # ============================================================
# # VALIDATE DATA
# # ============================================================

# def validate_data(resumes, jobs):
#     """Check that the required columns exist."""

#     required_resume_columns = {
#         "ID",
#         "Resume_str",
#         "Category",
#     }

#     required_job_columns = {
#         "Job_Title",
#         "Category",
#         "Job_Description",
#     }

#     missing_resume_columns = (
#         required_resume_columns
#         - set(resumes.columns)
#     )

#     missing_job_columns = (
#         required_job_columns
#         - set(jobs.columns)
#     )

#     if missing_resume_columns:
#         raise ValueError(
#             "Resume dataset is missing columns: "
#             + ", ".join(sorted(missing_resume_columns))
#         )

#     if missing_job_columns:
#         raise ValueError(
#             "Jobs dataset is missing columns: "
#             + ", ".join(sorted(missing_job_columns))
#         )


# # ============================================================
# # BUILD GROUND-TRUTH TEMPLATE
# # ============================================================

# def build_ground_truth(resumes, jobs):
#     """
#     Create a template for manual relevance annotation.

#     We only use categories to select candidate resumes and jobs.
#     Category matching is NOT treated as final relevance.
#     """

#     resume_categories = set(
#         resumes["Category"]
#         .dropna()
#         .astype(str)
#         .str.strip()
#     )

#     job_categories = set(
#         jobs["Category"]
#         .dropna()
#         .astype(str)
#         .str.strip()
#     )

#     common_categories = sorted(
#         resume_categories.intersection(job_categories)
#     )

#     rows = []

#     for category in common_categories:

#         category_resumes = resumes[
#             resumes["Category"]
#             .astype(str)
#             .str.strip()
#             == category
#         ].head(RESUMES_PER_CATEGORY)

#         category_jobs = jobs[
#             jobs["Category"]
#             .astype(str)
#             .str.strip()
#             == category
#         ]

#         candidate_job_titles = (
#             category_jobs["Job_Title"]
#             .dropna()
#             .astype(str)
#             .tolist()
#         )

#         candidate_jobs_text = "|".join(
#             candidate_job_titles
#         )

#         for _, resume in category_resumes.iterrows():

#             rows.append(
#                 {
#                     "resume_id": resume["ID"],
#                     "category": category,
#                     "candidate_jobs": candidate_jobs_text,
#                     "relevant_jobs": "",
#                     "review_status": "NEEDS_REVIEW",
#                 }
#             )

#     return pd.DataFrame(rows)


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print("=" * 70)
#     print("GROUND-TRUTH TEMPLATE GENERATOR")
#     print("=" * 70)

#     # --------------------------------------------------------
#     # Load
#     # --------------------------------------------------------

#     print("\nLoading datasets...")

#     resumes, jobs = load_data()

#     print(
#         f"Loaded {len(resumes)} resumes."
#     )

#     print(
#         f"Loaded {len(jobs)} jobs."
#     )

#     # --------------------------------------------------------
#     # Validate
#     # --------------------------------------------------------

#     validate_data(
#         resumes,
#         jobs,
#     )

#     # --------------------------------------------------------
#     # Build template
#     # --------------------------------------------------------

#     print(
#         "\nFinding categories that exist in both datasets..."
#     )

#     ground_truth = build_ground_truth(
#         resumes,
#         jobs,
#     )

#     if ground_truth.empty:

#         print(
#             "\nNo matching categories were found."
#         )

#         return

#     # --------------------------------------------------------
#     # Create output directory
#     # --------------------------------------------------------

#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     # --------------------------------------------------------
#     # Save
#     # --------------------------------------------------------

#     ground_truth.to_csv(
#         OUTPUT_PATH,
#         index=False,
#     )

#     # --------------------------------------------------------
#     # Display summary
#     # --------------------------------------------------------

#     print("\n" + "=" * 70)
#     print("GROUND-TRUTH TEMPLATE CREATED")
#     print("=" * 70)

#     print(
#         f"\nRows created: {len(ground_truth)}"
#     )

#     print(
#         f"Categories covered: "
#         f"{ground_truth['category'].nunique()}"
#     )

#     print(
#         "\nFile saved at:"
#     )

#     print(OUTPUT_PATH)

#     print(
#         "\nIMPORTANT:"
#     )

#     print(
#         "The relevant_jobs column is intentionally empty."
#     )

#     print(
#         "You must manually decide which candidate jobs "
#         "are relevant to each resume."
#     )

#     print(
#         "\nAfter deciding, change:"
#     )

#     print(
#         "NEEDS_REVIEW -> APPROVED"
#     )

#     print(
#         "\nDo NOT treat category matching as automatic "
#         "ground truth."
#     )


# if __name__ == "__main__":
#     main()
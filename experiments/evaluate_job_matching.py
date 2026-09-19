"""
Stratified evaluation of the hybrid job matching system.

This evaluation selects one usable resume from each category
so that the test set is not dominated by a single category.

Metrics:
    Precision@5
    Recall@5
    MRR

IMPORTANT:
The current ground truth uses category-based relevance.
Therefore, these are baseline evaluation metrics and should
not be described as model accuracy.
"""

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT MATCHER
# ============================================================

from backend.job_matcher import match_jobs


# ============================================================
# FILE PATHS
# ============================================================

RESUME_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "resume_dataset.csv"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "job_matching_ground_truth.csv"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "job_matching_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 5

# One resume from each category.
MAX_CATEGORIES = 21


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_relevant_jobs(value):
    """Convert pipe-separated job names into a set."""

    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    return {
        job.strip()
        for job in text.split("|")
        if job.strip()
    }


def get_resume_text(resume):
    """Safely get usable resume text."""

    value = resume.get(
        "Resume_str",
        ""
    )

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.lower() == "nan":
        return None

    return text


def precision_at_k(
    retrieved_jobs,
    relevant_jobs,
    k=5
):
    """Calculate Precision@K."""

    retrieved_at_k = retrieved_jobs[:k]

    if not retrieved_at_k:
        return 0.0

    relevant_count = sum(
        job in relevant_jobs
        for job in retrieved_at_k
    )

    return relevant_count / len(
        retrieved_at_k
    )


def recall_at_k(
    retrieved_jobs,
    relevant_jobs,
    k=5
):
    """Calculate Recall@K."""

    if not relevant_jobs:
        return None

    retrieved_at_k = retrieved_jobs[:k]

    relevant_count = sum(
        job in relevant_jobs
        for job in retrieved_at_k
    )

    return relevant_count / len(
        relevant_jobs
    )


def reciprocal_rank(
    retrieved_jobs,
    relevant_jobs
):
    """Calculate reciprocal rank."""

    for rank, job in enumerate(
        retrieved_jobs,
        start=1
    ):

        if job in relevant_jobs:
            return 1.0 / rank

    return 0.0


# ============================================================
# BUILD STRATIFIED SAMPLE
# ============================================================

def select_stratified_resumes(
    approved,
    resumes
):
    """
    Select one usable resume from each category.

    This prevents the evaluation from accidentally becoming:

        HR
        HR
        HR
        HR
        ...

    Instead we aim for:

        HR
        Finance
        Sales
        Engineering
        Healthcare
        ...
    """

    candidates = []

    for _, truth_row in approved.iterrows():

        resume_id = truth_row["resume_id"]

        resume_matches = resumes[
            resumes["ID"] == resume_id
        ]

        if resume_matches.empty:
            continue

        resume = resume_matches.iloc[0]

        resume_text = get_resume_text(
            resume
        )

        if resume_text is None:
            continue

        relevant_jobs = parse_relevant_jobs(
            truth_row["relevant_jobs"]
        )

        if not relevant_jobs:
            continue

        candidates.append(
            {
                "resume_id": resume_id,
                "category": resume["Category"],
                "resume_text": resume_text,
                "relevant_jobs": relevant_jobs
            }
        )

    candidates_df = pd.DataFrame(
        candidates
    )

    if candidates_df.empty:
        return []

    # Sort categories alphabetically so the selection
    # is deterministic.
    candidates_df = candidates_df.sort_values(
        by=["category", "resume_id"]
    )

    # Take ONE resume from each category.
    selected = (
        candidates_df
        .groupby(
            "category",
            sort=True,
            group_keys=False
        )
        .head(1)
    )

    # Limit to the configured number of categories.
    selected = selected.head(
        MAX_CATEGORIES
    )

    return selected.to_dict(
        orient="records"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STRATIFIED JOB MATCHING EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not RESUME_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Resume dataset not found:\n"
            f"{RESUME_DATA_PATH}"
        )

    if not GROUND_TRUTH_PATH.exists():

        raise FileNotFoundError(
            f"Ground truth not found:\n"
            f"{GROUND_TRUTH_PATH}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading datasets...")

    resumes = pd.read_csv(
        RESUME_DATA_PATH
    )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH
    )

    print(
        f"Total resumes: "
        f"{len(resumes)}"
    )

    print(
        f"Ground-truth rows: "
        f"{len(ground_truth)}"
    )

    # --------------------------------------------------------
    # Approved rows
    # --------------------------------------------------------

    approved = ground_truth[
        ground_truth["review_status"]
        .astype(str)
        .str.strip()
        .str.upper()
        == "APPROVED"
    ].copy()

    print(
        f"Approved rows: "
        f"{len(approved)}"
    )

    # --------------------------------------------------------
    # Select one resume per category
    # --------------------------------------------------------

    print()
    print(
        "Selecting one usable resume "
        "from each category..."
    )

    evaluation_rows = select_stratified_resumes(
        approved,
        resumes
    )

    if not evaluation_rows:

        print(
            "\nNo usable resumes found."
        )

        return

    print(
        f"\nSelected resumes: "
        f"{len(evaluation_rows)}"
    )

    print(
        "\nCategories selected:"
    )

    for item in evaluation_rows:

        print(
            f"  - {item['category']}"
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)

    for position, item in enumerate(
        evaluation_rows,
        start=1
    ):

        resume_id = item["resume_id"]

        category = item["category"]

        resume_text = item["resume_text"]

        relevant_jobs = item["relevant_jobs"]

        print()
        print(
            "-" * 70
        )

        print(
            f"[{position}/{len(evaluation_rows)}] "
            f"Resume ID: {resume_id}"
        )

        print(
            f"Category: {category}"
        )

        print(
            f"Relevant jobs: "
            f"{sorted(relevant_jobs)}"
        )

        # ----------------------------------------------------
        # Run matcher
        # ----------------------------------------------------

        try:

            matched_jobs = match_jobs(
                resume_text,
                top_n=TOP_K
            )

        except Exception as error:

            print(
                f"ERROR while matching "
                f"resume {resume_id}:"
            )

            print(error)

            continue

        # ----------------------------------------------------
        # Extract top jobs
        # ----------------------------------------------------

        if matched_jobs.empty:

            retrieved_jobs = []

        else:

            retrieved_jobs = (
                matched_jobs[
                    "Job_Title"
                ]
                .astype(str)
                .tolist()
            )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        precision = precision_at_k(
            retrieved_jobs,
            relevant_jobs,
            TOP_K
        )

        recall = recall_at_k(
            retrieved_jobs,
            relevant_jobs,
            TOP_K
        )

        reciprocal = reciprocal_rank(
            retrieved_jobs,
            relevant_jobs
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print(
            "\nTop 5 jobs:"
        )

        for rank, job in enumerate(
            retrieved_jobs,
            start=1
        ):

            marker = (
                "✓"
                if job in relevant_jobs
                else "✗"
            )

            print(
                f"  {rank}. "
                f"{job} {marker}"
            )

        print(
            f"\nPrecision@5: "
            f"{precision:.4f}"
        )

        print(
            f"Recall@5: "
            f"{recall:.4f}"
        )

        print(
            f"Reciprocal Rank: "
            f"{reciprocal:.4f}"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "resume_id": resume_id,
                "category": category,
                "relevant_jobs": "|".join(
                    sorted(relevant_jobs)
                ),
                "top_5_jobs": "|".join(
                    retrieved_jobs
                ),
                "precision_at_5": precision,
                "recall_at_5": recall,
                "reciprocal_rank": reciprocal
            }
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    if not results:

        print()
        print(
            "No evaluation results were generated."
        )

        return

    results_df = pd.DataFrame(
        results
    )

    mean_precision = (
        results_df[
            "precision_at_5"
        ].mean()
    )

    mean_recall = (
        results_df[
            "recall_at_5"
        ].mean()
    )

    mean_mrr = (
        results_df[
            "reciprocal_rank"
        ].mean()
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL STRATIFIED JOB MATCHING EVALUATION")
    print("=" * 70)

    print(
        f"\nCategories / resumes evaluated: "
        f"{len(results_df)}"
    )

    print(
        f"\nPrecision@5: "
        f"{mean_precision:.4f}"
    )

    print(
        f"Recall@5:    "
        f"{mean_recall:.4f}"
    )

    print(
        f"MRR:         "
        f"{mean_mrr:.4f}"
    )

    print(
        "\nDetailed results saved to:"
    )

    print(
        RESULTS_PATH
    )

    print()
    print("=" * 70)

    print(
        "\nNOTE:"
    )

    print(
        "These are category-based baseline metrics."
    )

    print(
        "They are not model accuracy."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
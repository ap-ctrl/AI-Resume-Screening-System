import json
import os

import pandas as pd

from backend.job_analyzer import analyze_job


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


JOBS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "jobs.csv"
)


CACHE_DIR = os.path.join(
    BASE_DIR,
    "cache"
)


CACHE_PATH = os.path.join(
    CACHE_DIR,
    "job_profiles.json"
)


# ============================================================
# LOAD JOBS
# ============================================================

def load_jobs():

    return pd.read_csv(
        JOBS_PATH
    )


# ============================================================
# CREATE JOB PROFILES
# ============================================================

def create_job_profiles():

    jobs_df = load_jobs()

    # --------------------------------------------------------
    # The cache folder should already exist.
    # We created it manually in the project root.
    # --------------------------------------------------------

    if not os.path.isdir(
        CACHE_DIR
    ):

        raise FileNotFoundError(
            f"Cache directory does not exist:\n"
            f"{CACHE_DIR}\n\n"
            f"Please create a folder named 'cache' "
            f"in the project root."
        )


    job_profiles = []


    total_jobs = len(
        jobs_df
    )


    print(
        f"Found {total_jobs} jobs."
    )


    print(
        "\nStarting job analysis...\n"
    )


    # ========================================================
    # ANALYZE EVERY JOB
    # ========================================================

    for index, row in jobs_df.iterrows():

        job_title = str(
            row["Job_Title"]
        )


        print(
            f"Analyzing job "
            f"{index + 1}/{total_jobs}: "
            f"{job_title}"
        )


        job_description = str(
            row["Job_Description"]
        )


        profile = analyze_job(
            job_description
        )


        job_profile = {

            "Job_Title": job_title,

            "Category": str(
                row["Category"]
            ),

            "Job_Description": job_description,

            "required_skills": profile[
                "required_skills"
            ],

            "preferred_skills": profile[
                "preferred_skills"
            ],

            "experience": profile[
                "experience"
            ],

            "education": profile[
                "education"
            ]
        }


        job_profiles.append(
            job_profile
        )


    # ========================================================
    # SAVE CACHE
    # ========================================================

    with open(
        CACHE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            job_profiles,
            file,
            indent=4,
            ensure_ascii=False
        )


    print(
        "\n========================================"
    )

    print(
        "JOB PROFILE CACHE CREATED"
    )

    print(
        "========================================"
    )


    print(
        f"Total jobs analyzed: "
        f"{len(job_profiles)}"
    )


    print(
        f"Cache saved at:"
    )


    print(
        CACHE_PATH
    )


    return job_profiles


# ============================================================
# LOAD CACHED PROFILES
# ============================================================

def load_cached_job_profiles():

    if not os.path.isfile(
        CACHE_PATH
    ):

        return None


    with open(
        CACHE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# GET OR CREATE CACHE
# ============================================================

def get_job_profiles():

    cached_profiles = (
        load_cached_job_profiles()
    )


    # --------------------------------------------------------
    # If cache already exists, don't call Groq again.
    # --------------------------------------------------------

    if cached_profiles is not None:

        print(
            "Existing job profile cache found."
        )

        print(
            f"Loaded "
            f"{len(cached_profiles)} "
            f"cached job profiles."
        )

        return cached_profiles


    # --------------------------------------------------------
    # Otherwise analyze all jobs.
    # --------------------------------------------------------

    print(
        "No existing job profile cache found."
    )

    print(
        "Creating job profiles using Groq..."
    )


    return create_job_profiles()


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    profiles = get_job_profiles()


    print(
        f"\nLoaded "
        f"{len(profiles)} job profiles."
    )


    if profiles:

        print(
            "\nFirst job profile:"
        )


        print(
            json.dumps(
                profiles[0],
                indent=4,
                ensure_ascii=False
            )
        )
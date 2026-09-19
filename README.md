# AI Resume Screening & Job Matching System

An AI-powered resume screening and job matching system that analyzes resumes, extracts structured candidate information, identifies skills, and matches candidates with relevant job opportunities using a combination of Large Language Models (LLMs), Natural Language Processing (NLP), semantic similarity, and skill-based matching.

The system is designed to provide not only job matches but also explainable information about why a particular job matches a candidate's profile and which skills are exact, related, or missing.

---

## Features

### 1. Multi-Format Resume Input

The system supports resumes in multiple formats:

- PDF
- DOCX
- TXT
- Pasted resume text

The resume is first converted into clean text before further analysis.

---

### 2. AI-Powered Resume Analysis

A Groq-powered LLM analyzes the extracted resume text and generates a structured candidate profile.

The profile can contain:

- Professional summary
- Technical skills
- Soft skills
- Work experience
- Education
- Projects
- Certifications
- Other relevant candidate information

This allows unstructured resume text to be converted into structured information that can be used by the matching system.

---

### 3. Skill Normalization

Different ways of writing the same skill are normalized into a common representation.

For example:

- `ML` → `Machine Learning`
- `AI` → `Artificial Intelligence`
- `NLP` → `Natural Language Processing`
- `sklearn` → `Scikit-learn`

This improves consistency during job matching.

---

### 4. Job Profile Extraction

Job descriptions are analyzed using the LLM to identify:

- Required skills
- Preferred skills
- Experience requirements
- Education requirements

The extracted job profiles are stored in a local cache so that the system does not need to repeatedly analyze the same job descriptions.

---

### 5. Hybrid Job Matching

The system combines multiple matching techniques instead of relying on a single similarity method.

The final matching score combines:

- TF-IDF similarity
- Semantic similarity
- Required skill coverage
- Preferred skill coverage

The current scoring approach is:

```text
Final Score =
0.20 × TF-IDF Similarity
+ 0.30 × Semantic Similarity
+ 0.40 × Required Skill Coverage
+ 0.10 × Preferred Skill Coverage
```

This hybrid approach combines traditional NLP, semantic understanding, and explicit skill matching.

---

### 6. Semantic Matching

The system uses the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

to generate semantic embeddings for resume and job-description text.

This allows the system to identify similarities even when the wording is not exactly the same.

For example:

```text
"Data visualization using Power BI"
```

and:

```text
"Creating business dashboards and visual reports"
```

may have semantic similarity even though the wording differs.

---

### 7. Exact, Related and Missing Skills

The matching system provides more detailed skill information instead of only producing a single similarity score.

Skills are divided into:

#### Exact Skills

Skills directly present in both the candidate profile and job requirements.

Example:

```text
Python
SQL
Machine Learning
```

#### Related Skills

Skills where the candidate has a related technology or concept.

Example:

```text
Cloud Deployment ← Docker
```

This indicates that Docker provides related evidence for cloud/deployment requirements, but is not treated as an exact match.

#### Missing Skills

Required skills that could not be matched from the candidate profile.

Example:

```text
Backend Development
System Design
```

This makes the matching results more explainable.

---

### 8. Explainable Job Matching

For each matched job, the system can generate an explanation using the Groq LLM.

The explanation is grounded in the existing candidate and job information.

It provides sections such as:

- Why this job matches
- Strong matches
- Related evidence
- Main skill gaps

The explanation does not independently recalculate the matching score or invent additional candidate skills.

---

## Why Use Both LLM and Traditional ML/NLP?

The system uses different techniques for different parts of the problem.

### LLM

The LLM is useful for understanding unstructured natural language.

It is used for:

- Resume information extraction
- Job description analysis
- Skill extraction
- Structured profile generation
- Natural-language explanations

### Traditional NLP / Machine Learning

Traditional techniques provide measurable and reproducible matching signals.

They are used for:

- TF-IDF similarity
- Semantic similarity
- Explicit skill matching
- Required skill coverage
- Preferred skill coverage
- Resume classification experiments

Using both approaches creates a hybrid system where the LLM handles language understanding while deterministic and ML-based techniques contribute measurable matching signals.

---

## System Architecture

```text
                    Resume
                      │
          ┌───────────┴───────────┐
          │ PDF / DOCX / TXT      │
          │ or Pasted Text        │
          └───────────┬───────────┘
                      │
                      ▼
                Resume Parser
                      │
                      ▼
              Groq Resume Analyzer
                      │
                      ▼
          Structured Candidate Profile
                      │
                      ▼
              Skill Normalization
                      │
                      ▼
              Hybrid Job Matching
          ┌───────────┼───────────┐
          │           │           │
        TF-IDF     Semantic     Skills
        Similarity Similarity   Matching
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
                 Top Jobs
                      │
                      ▼
          Exact / Related / Missing
                      │
                      ▼
             Groq Explanation
                      │
                      ▼
             Explainable Results
```

---

## Project Structure

```text
AI-Resume-Screening-System/
│
├── backend/
│   ├── groq_client.py
│   ├── job_analyzer.py
│   ├── job_matcher.py
│   ├── job_profile_cache.py
│   ├── llm_explainer.py
│   ├── predict.py
│   ├── resume_analyzer.py
│   ├── resume_parser.py
│   ├── semantic_matcher.py
│   ├── skill_gap.py
│   ├── skill_matcher.py
│   └── skill_normalizer.py
│
├── frontend/
│   └── app.py
│
├── data/
│   ├── jobs.csv
│   └── resume_dataset.csv
│
├── models/
│   ├── resume_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── cache/
│   └── job_profiles.json
│
├── evaluation/
│   ├── job_matching_ground_truth.csv
│   └── job_matching_results.csv
│
├── experiments/
│   ├── advanced_preprocess.py
│   ├── check_data.py
│   ├── confusion_matrix_analysis.py
│   ├── evaluate_job_matching.py
│   ├── generate_ground_truth.py
│   ├── model_training.py
│   ├── model_training_svc.py
│   ├── model_training_svc_v2.py
│   ├── model_training_v2.py
│   ├── preprocess.py
│   ├── train_and_save_model.py
│   ├── svc_tuning.py
│   └── vectorize.py
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

### Programming Language

- Python

### Frontend

- Streamlit

### LLM

- Groq API

### NLP / Machine Learning

- Scikit-learn
- TF-IDF
- Sentence Transformers
- Cosine Similarity

### Embeddings

```text
all-MiniLM-L6-v2
```

### Data Processing

- Pandas
- NumPy

### Resume Processing

- PyPDF2
- python-docx

### Model / File Handling

- Joblib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ap-ctrl/AI-Resume-Screening-System.git
```

Move into the project directory:

```bash
cd AI-Resume-Screening-System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Groq API Key Setup

The application requires a Groq API key for LLM-based resume analysis, job analysis, and explanations.

Create an environment variable named:

```text
GROQ_API_KEY
```

For local development, a `.env` file can be used:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Do not commit the `.env` file or expose the API key publicly.

---

## Running the Application

From the project root directory, run:

```bash
streamlit run frontend/app.py
```

The Streamlit application will open in your browser.

---

## Application Workflow

### Step 1: Upload Resume

The user can upload:

- PDF
- DOCX
- TXT

or paste resume text directly.

---

### Step 2: Resume Parsing

The system extracts readable text from the uploaded resume.

---

### Step 3: Resume Analysis

The extracted text is sent to the Groq-powered analyzer.

The analyzer creates a structured candidate profile containing information such as:

```text
Summary
Technical Skills
Soft Skills
Experience
Education
Projects
Certifications
```

---

### Step 4: Skill Normalization

Extracted skills are normalized to improve consistency.

For example:

```text
ML
Machine Learning
machine-learning
```

can be treated as:

```text
Machine Learning
```

---

### Step 5: Job Matching

The candidate profile is compared against available job descriptions.

The system calculates:

```text
TF-IDF Similarity
Semantic Similarity
Required Skill Coverage
Preferred Skill Coverage
```

These signals are combined into the final matching score.

---

### Step 6: Skill Breakdown

For each job, the system shows:

```text
Exact Skills
Related Skills
Missing Skills
```

This helps explain the differences between the candidate's profile and the job requirements.

---

### Step 7: Job Explanation

The user can request an explanation for a matched job.

The Groq LLM generates a grounded explanation based on:

- Candidate skills
- Job requirements
- Exact matches
- Related matches
- Missing skills

---

## Job Profile Caching

Job descriptions are analyzed and converted into structured job profiles.

The generated profiles are stored in:

```text
cache/job_profiles.json
```

This avoids repeatedly sending the same job descriptions to the LLM.

To generate or refresh the job profile cache, run:

```bash
python -m backend.job_profile_cache
```

---

## Matching Methodology

The matching system uses four major components.

### TF-IDF Similarity

Measures lexical similarity between resume text and job-description text.

It is useful when the same or similar terms occur in both documents.

---

### Semantic Similarity

Sentence Transformers are used to create embeddings.

The system compares the semantic representations of the resume and job description.

This helps identify meaning-level similarity beyond exact keywords.

---

### Required Skill Coverage

Required job skills have the highest contribution among the skill-based components.

Matching is calculated using:

```text
Exact Match = 1.0
Related Match = 0.5
Missing Match = 0.0
```

---

### Preferred Skill Coverage

Preferred skills contribute to the final score but have lower weight than required skills.

---

## Evaluation

A baseline evaluation was performed using one usable resume from each available resume category.

The evaluation used:

```text
21 resume categories
```

The current baseline results were:

```text
Precision@5: 20.00%

Recall@5: 88.89%

MRR: 0.7016
```

### Metric Meaning

#### Precision@5

Measures how many of the top 5 retrieved jobs are considered relevant.

#### Recall@5

Measures how many relevant jobs were retrieved within the top 5 results.

#### MRR

Mean Reciprocal Rank measures how highly the first relevant result appears in the ranked list.

---

## Evaluation Limitation

The current evaluation uses automatically generated category-based relevance labels.

The ground truth was generated using the relationship:

```text
Resume Category == Job Category
```

Therefore, these results should be treated as a **category-based baseline evaluation**, not as human-annotated ground truth.

The evaluation metrics should not be interpreted as overall model accuracy.

A stronger future evaluation would use manually reviewed resume-job relevance labels from domain experts or multiple human annotators.

---

## Existing ML Classification Experiments

The project also contains earlier experiments for resume classification using traditional machine learning and NLP techniques.

These experiments include:

- Text preprocessing
- TF-IDF vectorization
- Resume classification
- Support Vector Classification
- Hyperparameter tuning
- Confusion matrix analysis

These experiments are maintained separately from the current hybrid job-matching pipeline.

---

## Limitations

The current system has several limitations:

- Job matching quality depends on the quality of the resume text and job descriptions.
- LLM-generated extraction can occasionally miss or incorrectly interpret information.
- Related-skill mappings are currently based on manually defined relationships.
- The current evaluation uses category-based automatically generated ground truth.
- The semantic model is a general-purpose embedding model rather than a job-recruitment-specific model.
- The system requires a Groq API key for LLM functionality.
- API usage may be subject to provider limits and availability.
- The current system does not perform full applicant tracking system integration.

---

## Future Improvements

Possible future improvements include:

- Human-annotated evaluation datasets
- More comprehensive skill ontologies
- Better skill relationship detection
- Domain-specific embedding models
- Improved job ranking models
- Experience-level matching
- Education requirement matching
- Location and work-mode matching
- Salary compatibility matching
- Improved evaluation methodology
- Candidate-job feedback loops
- More robust handling of resume formatting
- Support for additional document formats

---

## Key Design Principle

The system does not rely on an LLM alone.

Instead, it uses a hybrid architecture:

```text
LLM
↓
Understand and structure language

Traditional NLP / ML
↓
Measure similarity

Skill Matching
↓
Provide explicit evidence

LLM
↓
Explain the results
```

This separation helps make the system more measurable and explainable while still taking advantage of modern language-model capabilities.

---

## Author

Developed as an AI/ML project focused on resume screening, NLP, semantic search, and job matching.

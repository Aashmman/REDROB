
import json
from utils.normalization import normalize_skills
from utils.deduplication import deduplicate_skills
from utils.tf_idf import calculate_tfidf
from utils.cosine_similarity import calculate_cosine_similarity


# Load data
with open('data/resumes.json') as f:
    resumes = json.load(f)

with open('data/job_descriptions.json') as f:
    job_descriptions = json.load(f)

with open('data/skill_aliases.json') as f:
    skill_aliases = json.load(f)


# Normalize skills
normalized_resumes = []

for resume in resumes:
    normalized_skills = normalize_skills(
        resume['skills'],
        skill_aliases
    )

    resume['skills'] = normalized_skills
    normalized_resumes.append(resume)


# Deduplicate skills
deduplicated_resumes = []

for resume in normalized_resumes:
    deduplicated_skills = deduplicate_skills(
        resume['skills']
    )

    resume['skills'] = deduplicated_skills
    deduplicated_resumes.append(resume)


# Calculate TF-IDF vectors
tfidf_vectors = calculate_tfidf(
    deduplicated_resumes
)


# Calculate cosine similarity
cosine_similarities = calculate_cosine_similarity(
    tfidf_vectors,
    job_descriptions
)


# Output results
with open('output/results.txt', 'w') as f:

    for job_description in job_descriptions:

        f.write(
            f"Job Description: "
            f"{job_description['title']}\n"
        )

        for resume in cosine_similarities[
            job_description['title']
        ]:

            f.write(
                f"Resume: {resume['name']}, "
                f"Score: {resume['score']:.2f}\n"
            )

        f.write("\n")


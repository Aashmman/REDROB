
import json
import os

from utils.normalization import normalize_skills
from utils.deduplication import deduplicate_skills
from utils.tf_idf import calculate_tfidf
from utils.cosine_similarity import calculate_cosine_similarity


# Load resume data
with open('data/resumes.json', 'r') as f:
    resumes = json.load(f)


# Load job descriptions
with open('data/job_descriptions.json', 'r') as f:
    job_descriptions = json.load(f)


# Load skill aliases
with open('data/skill_aliases.json', 'r') as f:
    skill_aliases = json.load(f)


# Normalize resume skills
normalized_resumes = []

for resume in resumes:

    normalized_skills = normalize_skills(
        resume['skills'],
        skill_aliases
    )

    resume['skills'] = normalized_skills

    normalized_resumes.append(resume)


# Deduplicate resume skills
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


# Calculate cosine similarities
cosine_similarities = calculate_cosine_similarity(
    tfidf_vectors,
    job_descriptions
)


# Create output folder if missing
os.makedirs("output", exist_ok=True)


# Write results
with open("output/results.txt", "w") as f:

    for job_description in job_descriptions:

        f.write(
            f"{job_description['title']}\n"
        )

        results = cosine_similarities[
            job_description['title']
        ]

        # Sort by score descending, then name ascending
        results = sorted(
            results,
            key=lambda x: (-x['score'], x['name'])
        )

        top_3 = results[:3]

        formatted_results = []

        for candidate in top_3:

            formatted_results.append(
                f"{candidate['name']}("
                f"{candidate['score']:.2f})"
            )

        f.write(
            ", ".join(formatted_results)
        )

        f.write("\n\n")


print("Results generated successfully!")
print("Check output/results.txt")


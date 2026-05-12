
import math

from utils.normalization import normalize_skills


def calculate_cosine_similarity(
    tfidf_vectors,
    job_descriptions,
    skill_aliases
):

    cosine_similarities = {}

    for job_description in job_descriptions:

        # Normalize JD skills
        required_skills = normalize_skills(
            job_description['required_skills'],
            skill_aliases
        )

        job_description_vector = {}

        for skill in required_skills:

            job_description_vector[skill] = 1

        cosine_similarities[
            job_description['title']
        ] = []

        for resume in tfidf_vectors:

            dot_product = 0

            for skill in resume['vector']:

                if skill in job_description_vector:

                    dot_product += (
                        resume['vector'][skill]
                        * job_description_vector[skill]
                    )

            resume_norm = math.sqrt(
                sum(
                    x ** 2
                    for x in resume['vector'].values()
                )
            )

            jd_norm = math.sqrt(
                sum(
                    x ** 2
                    for x in job_description_vector.values()
                )
            )

            if resume_norm == 0 or jd_norm == 0:
                similarity = 0
            else:
                similarity = dot_product / (
                    resume_norm * jd_norm
                )

            cosine_similarities[
                job_description['title']
            ].append({
                'name': resume['name'],
                'score': similarity
            })

    return cosine_similarities



import math


def calculate_tfidf(resumes):

    tfidf_vectors = []

    for resume in resumes:

        tfidf_vector = {}

        total_skills = len(resume['skills'])

        for skill in resume['skills']:

            tf = 1 / total_skills

            document_frequency = len([
                r for r in resumes
                if skill in r['skills']
            ])

            idf = math.log(10 / document_frequency)

            tfidf_vector[skill] = tf * idf

        tfidf_vectors.append({
            'name': resume['name'],
            'vector': tfidf_vector
        })

    return tfidf_vectors
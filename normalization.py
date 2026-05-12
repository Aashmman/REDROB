
import json


def normalize_skills(skills, skill_aliases):

    normalized_skills = []

    for skill in skills:

        skill = skill.lower().strip()

        if skill in skill_aliases:

            normalized_skills.append(
                skill_aliases[skill]
            )

    return normalized_skills

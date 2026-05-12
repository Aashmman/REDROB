def deduplicate_skills(skills):
    deduplicated_skills = []
    for skill in skills:
        if skill not in deduplicated_skills:
            deduplicated_skills.append(skill)
    return deduplicated_skills
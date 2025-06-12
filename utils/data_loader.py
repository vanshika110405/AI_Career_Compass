import pandas as pd

def load_careers(csv_path):
    df = pd.read_csv(csv_path)

    # Drop duplicate rows based on the 'role' column (keep first occurrence)
    df = df.drop_duplicates(subset='role', keep='first')

    return df.to_dict(orient='records')


def search_careers(careers, query):
    query = query.lower()
    results = []
    for career in careers:
        if query in career['role'].lower() or query in career['description'].lower():
            results.append(career)
        else:
            # Check in skills or industries (comma separated fields)
            if query in career['required_skills'].lower() or query in career['domain_industries'].lower():
                results.append(career)
    return results

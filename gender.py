import praw
import pandas as pd
import re
import spacy
import matplotlib.pyplot as plt
from NER import NamedEntityRecognition
from medspacy.ner import TargetMatcher, TargetRule
from scispacy.linking import EntityLinker


def extract_names(umls_links):
    """
    Extracts unique 'name' fields from UMLS links.
    """
    try:
        names = []
        for entity in umls_links:
            if isinstance(entity, dict) and entity.get('umls_links'):
                names.extend(link.get('name') for link in entity['umls_links'] if link.get('name'))
        return list(set(names)) 
    except Exception as e:
        print(f"Error parsing umls_links: {e}")
        return []


def extract_gender_age(text):
    """
    Extracts gender and age from text using regex patterns.
    """
    match = re.search(r'\b((?:[MF]\d{1,3}|\d{1,3}[MF]))\b', text)
    if match:
        extracted = match.group(1)
        gender = extracted[0].upper() if extracted[0] in 'MF' else extracted[-1].upper()
        age = extracted[1:] if gender == extracted[0] else extracted[:-1]
        return gender, int(age)
    return None, None


def process_posts(posts, nlp):
    """
    Processes Reddit posts to extract gender, age, and symptoms.
    """
    data = []
    for post in posts:
        text = post.selftext
        entities = NamedEntityRecognition(text, nlp)

        symptoms = [
            link['name'] for entity in entities if entity.get('umls_links') for link in entity['umls_links']
            if link['name'].lower() not in {'symptom', 'other symptom', 'symptoms'}
        ]

        gender, age = extract_gender_age(text)

        if gender and age and symptoms:
            data.append({'text': text, 'gender': gender, 'age': age, 'symptoms': symptoms})

    return pd.DataFrame(data)


def analyze_age_distribution(df):
    """
    Analyzes and visualizes symptoms by age group.
    """
    bins = [0, 18, 30, 45, 60, 75, 100]
    labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
    df['age_range'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    print("Number of samples per age group:", df['age_range'].value_counts(sort=False))

    df_exploded = df.explode('symptoms')
    age_symptoms = df_exploded.groupby(['age_range', 'symptoms']).size().unstack(fill_value=0)

    top_symptoms = df_exploded['symptoms'].value_counts().head(10).index
    filtered_age_symptoms = age_symptoms.reindex(columns=top_symptoms, fill_value=0)

    plot_symptom_distribution(filtered_age_symptoms, "Age Group", "Top 10 Symptoms by Age Group")


def analyze_gender_distribution(df):
    """
    Analyzes and visualizes symptoms by gender.
    """
    df_exploded = df.explode('symptoms')

    total_male_posts = df[df['gender'] == 'M'].shape[0]
    total_female_posts = df[df['gender'] == 'F'].shape[0]
    print("Number of male posts:", total_male_posts)
    print("Number of female posts:", total_female_posts)

    male_symptoms = df_exploded[df_exploded['gender'] == 'M']['symptoms'].value_counts().head(10)
    female_symptoms = df_exploded[df_exploded['gender'] == 'F']['symptoms'].value_counts().head(10)

    all_symptoms = pd.DataFrame({'Male': male_symptoms, 'Female': female_symptoms}).fillna(0)

    plot_symptom_distribution(all_symptoms, "Symptoms", "Top 10 Symptoms by Gender")


def plot_symptom_distribution(data, xlabel, title):
    """
    Plots symptom distribution.
    """
    plt.figure(figsize=(14, 8))
    data.plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category')
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to fetch Reddit posts and analyze symptoms.
    """
    # Set up Reddit API
    reddit = praw.Reddit(
        client_id='REDDIT_CLIENT_ID', # Replace with your Reddit client ID
        client_secret='REDDIT_CLIENT_SECRET', # Replace with your Reddit client secret
        user_agent='Any_Acanthaceae915'
    )

    subreddit = reddit.subreddit('AskDocs')
    recent_posts = list(subreddit.new(limit=1000))

    # Load NLP model
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

    # Process posts and analyze
    df = process_posts(recent_posts, nlp)
    print(f"Number of processed posts: {len(df)}")

    analyze_age_distribution(df)
    analyze_gender_distribution(df)


if __name__ == "__main__":
    main()

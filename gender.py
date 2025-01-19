import praw
import pandas as pd
import re
from NER import NamedEntityRecognition
import spacy

import pandas as pd
from NER import NamedEntityRecognition
import matplotlib.pyplot as plt
from collections import Counter

import spacy
from medspacy.ner import TargetMatcher, TargetRule
from scispacy.linking import EntityLinker

def extract_names(umls_links):
    try:
        # Parse entities to extract unique 'name' fields
        names = []
        for entity in umls_links:
            if isinstance(entity, dict) and entity.get('umls_links'):
                for link in entity['umls_links']:
                    name = link.get('name')
                    if name:
                        names.append(name)
        return list(set(names))  # Return unique names
    except Exception as e:
        print(f"Error parsing umls_links: {e}")
        return []


def extract_gender_age(text):
    match = re.search(r'\b((?:[MF]\d{1,3}|\d{1,3}[MF]))\b', text)
    if match:
        extracted = match.group(1)
        if extracted[0] in 'MF':  #If gender comes first
            gender = extracted[0].upper()
            age = extracted[1:]
        else:  #If age comes first
            gender = extracted[-1].upper()
            age = extracted[:-1]
        return gender, int(age)
    return None, None

def analyze_gender_age_symptoms(posts, nlp):
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
            data.append({
                'text': text,
                'gender': gender,
                'age': age,
                'symptoms': symptoms
            })

    df = pd.DataFrame(data)
    print("Number of approved samples: ", len(df))

    bins = [0, 18, 30, 45, 60, 75, 100] 
    labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
    df['age_range'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    age_group_counts = df['age_range'].value_counts(sort=False)  
    print("Number of samples per age group:", age_group_counts)

    df_exploded = df.explode('symptoms')

    ###AGE ANALYSIS###
    age_symptoms = df_exploded.groupby(['age_range', 'symptoms']).size().unstack(fill_value=0)

    age_totals = age_symptoms.sum(axis=1)  
    normalized_age_symptoms = age_symptoms.div(age_totals, axis=0) * 100  

    top_symptoms = df_exploded['symptoms'].value_counts().head(10).index
    filtered_age_symptoms = age_symptoms.reindex(columns=top_symptoms, fill_value=0)
    filtered_normalized_age_symptoms = normalized_age_symptoms.reindex(columns=top_symptoms, fill_value=0)

    plt.rcParams.update({'font.size': 16})


    filtered_age_symptoms.plot(kind='bar', figsize=(14, 8))
    plt.title('Top 10 Symptoms by Age Group (Absolute Counts)')
    plt.xlabel('Age Group')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0, ha='center')
    plt.legend(title='Symptoms', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    filtered_normalized_age_symptoms.plot(kind='bar', figsize=(14, 8))
    plt.title('Top 10 Symptoms by Age Group (Normalized Percentages)')
    plt.xlabel('Age Group')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0, ha='center')
    plt.legend(title='Symptoms', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    ###GENDER ANALYSIS###
    total_male_posts = df[df['gender'] == 'M'].shape[0]
    total_female_posts = df[df['gender'] == 'F'].shape[0]

    print("Number of female posts: ", total_female_posts)
    print("Number of male posts: ", total_male_posts)

    male_symptoms = df_exploded[df_exploded['gender'] == 'M']['symptoms'].value_counts().head(10)
    female_symptoms = df_exploded[df_exploded['gender'] == 'F']['symptoms'].value_counts().head(10)

    normalized_male_symptoms = male_symptoms / total_male_posts
    normalized_female_symptoms = female_symptoms / total_female_posts

    all_symptoms = pd.DataFrame({
        'Male': male_symptoms,
        'Female': female_symptoms
    }).fillna(0)

    normalized_all_symptoms = pd.DataFrame({
        'Male': normalized_male_symptoms,
        'Female': normalized_female_symptoms
    }).fillna(0)

    all_symptoms.plot(kind='bar', figsize=(14, 8))
    plt.title(f'Top 10 Symptoms by Gender (Absolute Counts)')
    plt.xlabel('Symptoms')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.show()

    normalized_all_symptoms.plot(kind='bar', figsize=(14, 8))
    plt.title(f'Top 10 Symptoms by Gender (Normalized Percentages)')
    plt.xlabel('Symptoms')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #Set up Reddit API
    reddit = praw.Reddit(
        client_id='JjA1KIDbvcj5dmByl9Ei5Q',
        client_secret='LyX9AWEVTP3nUb9c_eHClKWVIdh1qg',
        user_agent='Any_Acanthaceae915'
    )

    subreddit = reddit.subreddit('AskDocs')
    # all_posts = []

    
    subreddit = reddit.subreddit('AskDocs')
    recent_posts = list(subreddit.new(limit=1000)) 
    
    data = []
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

    analyze_gender_age_symptoms(recent_posts, nlp)

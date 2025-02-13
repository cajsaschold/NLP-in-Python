# TDDE16

The code and report from the course Text Mining (TDDE16) given at Linköping University. The project analyzes gender and age differences in symptom descriptions from Reddit's AskDocs subreddit using Natural Language Processing (NLP) techniques.

The aim of this project is to investigate whether certain symptoms are reported more frequently by different genders and age groups in an online medical forum. The analysis is performed using Named Entity Recognition (NER) and concept linking with the UMLS medical database. Data is collected from the AskDocs subreddit using the Reddit API. Symptoms are extracted using SpaCy and ScispaCy, linking them to UMLS medical concepts.

## Files

gender.py - Extracts gender, age, and symptoms from Reddit posts and performs statistical analysis.

NER.py - Implements Named Entity Recognition (NER) to extract medical entities from text.

Text_Mining_Report.pdf - The final project report detailing the methodology, results, and conclusions.

## Requirements

**To run this project, install the following dependencies:**

pip install praw spacy scispacy matplotlib pandas

python -m spacy download en_core_sci_sm



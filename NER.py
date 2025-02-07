import re

def preprocess_text(text):
    """
    Preprocesses text by converting to lowercase, removing special characters, 
    and normalizing spaces.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  
    return " ".join(text.split())  


def extract_umls_links(ent, nlp, relevant_tuis):
    """
    Extracts UMLS links from a named entity, filtering by relevant TUIs.
    """
    if not hasattr(ent._, "kb_ents") or not ent._.kb_ents:
        return None

    linker = nlp.get_pipe("scispacy_linker") 
    umls_links = []

    for cui, score in ent._.kb_ents:
        entity = linker.kb.cui_to_entity.get(cui)
        if entity and any(tui in relevant_tuis for tui in entity.types):
            umls_links.append({
                "cui": cui,
                "name": entity.canonical_name,
                "types": entity.types,
                "similarity": score
            })

    return umls_links if umls_links else None


def NamedEntityRecognition(text, nlp):
    """
    Performs Named Entity Recognition (NER) on input text using the provided NLP model.
    Extracts entities relevant to the predefined TUIs (Type Unique Identifiers).
    """
    relevant_tuis = {"T184"}  # Modify to extract relevant UMLS types, list of UMLS can be found at https://gist.github.com/joelkuiper/4869d148333f279c2b2e
    entities = []  

    text = preprocess_text(text)
    doc = nlp(text)

    for ent in doc.ents:
        entity_data = {
            "text": ent.text,
            "label": ent.label_,
            "umls_links": extract_umls_links(ent, nlp, relevant_tuis)
        }

        if entity_data["umls_links"]:
            entities.append(entity_data)

    return entities

import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  
    text = " ".join(text.split()) 
    return text

def NamedEntityRecognition(text, nlp):

    relevant_tuis = {"T184"}

    entities = []  

    text = preprocess_text(text)

    doc = nlp(text)

    for ent in doc.ents:
        entity_data = {
            "text": ent.text,
            "label": ent.label_,
            "umls_links": []  
        }

        if hasattr(ent._, "kb_ents") and ent._.kb_ents:
            for umls_ent in ent._.kb_ents:
                cui = umls_ent[0]
                score = umls_ent[1]
                linker = nlp.get_pipe("scispacy_linker")
                entity = linker.kb.cui_to_entity[cui]
                if any(tui in relevant_tuis for tui in entity.types):
                    entity_data["umls_links"].append({
                        "cui": cui,
                        "name": entity.canonical_name,
                        "types": entity.types,
                        "similarity": score
                    })
        else:
            entity_data["umls_links"] = None  

        if entity_data["umls_links"]:
            entities.append(entity_data)
            

    return entities
    

# List of all TUIs
# https://gist.github.com/joelkuiper/4869d148333f279c2b2e




import spacy
from spacy.matcher import PhraseMatcher

# Load pretrained model
nlp = spacy.load("en_core_web_sm")


def process_to_doc(user_input):
    # Process the input with spaCy
    return nlp(user_input)


def nlp_extract_details(user_input, event_desc):
    print("from extract_event/extract_details: ", user_input)
    doc = process_to_doc(user_input)
    # Initialize variables to store extracted date, time, and event description
    dates = []
    times = []
    locations = []

    for ent in doc.ents:
        print(ent.label_)

    # Iterate over the entities recognized by spaCy
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.append(ent)
        elif ent.label_ == "TIME":
            times.append(ent)
        elif ent.label_ in ["GPE", "FAC", "ORG"]:
            # probably better to use custom logic to extract location by
            # if PREP is in front of an ORG or NAME
            locations.append(ent)

    # Further processing to clean up the event description
    # This can include removing extra spaces, handling conjunctions, etc.
    # For simplicity, this step is not detailed here

    return {
        "dates": [ent.text for ent in dates],
        "times": [ent.text for ent in times],
        "locations": [ent.text for ent in locations],
        "summary": event_desc,
        "description": event_desc,
    }


def nlp_understand_intent(user_input):
    print("from extract_event: ", user_input)
    doc = process_to_doc(user_input)
    matcher = PhraseMatcher(nlp.vocab)
    add_terms = [
        "add calendar",
        "add appointment",
        "add event",
        "add an event",
        "schedule a",
        "schedule an",
        "book a",
        "book an",
        "create a",
        "create an",
    ]
    # Only run nlp.make_doc to speed things up
    patterns = [nlp.make_doc(text) for text in add_terms]
    matcher.add("create_event", patterns)
    matches = matcher(doc)
    for matches_id, start, end in matches:
        span = doc[start:end]
        if matches_id and span:
            return "create_event"
        return None

import spacy
import json
import random
from spacy.training import Example

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Add the new entity label to the 'ner' pipeline
ner = nlp.get_pipe("ner")
ner.add_label("PLACES")
ner.add_label("ADDRESS")

# Disable other pipeline components for training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()

    # Load the training data from the JSON file
    # start/end position must match between the entity spans match the token boundaries
    # can use a for loop, ie
    # ### for token in nlp.make_doc("I visited the Eiffel Tower last month":
    # ####### print(token.text, token.idx, token.idx + len(token.text))

    with open("training_data.json", "r", encoding="utf-8") as f:
        TRAIN_DATA = json.load(f)

    # Train the model
    for itn in range(30):  # Number of training iterations
        random.shuffle(TRAIN_DATA)
        losses = {}
        for item in TRAIN_DATA:
            text = item["text"]
            annotations = item["annotations"]
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            print(example)
            nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
        print(losses)


# Save the updated model
nlp.to_disk("updated_model")

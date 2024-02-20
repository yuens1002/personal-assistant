from spacy_llm.util import assemble
from thinc.api import set_gpu_allocator, require_gpu

set_gpu_allocator("pytorch")
require_gpu(0)

nlp = assemble("_config.cfg")
doc = nlp("Jack and Jill rode up the hill in Les Deux Alpes")
print([(ent.text, ent.label_) for ent in doc.ents])

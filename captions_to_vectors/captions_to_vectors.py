import gensim
from os import listdir
from os.path import isfile, join
from collections import namedtuple


documents_path = '../data/pascal-sentences/'

train_data_path = documents_path + 'ps_captions_train/'
test_data_path = documents_path + 'ps_captions_test/'
valid_data_path = documents_path + 'ps_captions_valid/'

documents_train = []

train_data = [f for f in listdir(train_data_path) if isfile(join(train_data_path, f))]


namedtupleDocument = namedtuple('namedtupleDocument', 'doc name')
for document in train_data:
    with open(train_data_path+document, 'r') as f:
        doc = f.read()
        documents_train.append(namedtupleDocument(doc, document.split('.')[0]))

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for document in documents_train:
    words = document[0].lower().split()
    tags = [document[1]]
    docs.append(analyzedDocument(words, tags))

model = gensim.models.Doc2Vec(docs, size = 4000, window = 8, min_count = 5, workers = 4)

print(model.docvecs[0], model.docvecs.index_to_doctag(0))
# model.save('model_with_training_data')






import gensim
from os import listdir
from os.path import isfile, join
from collections import namedtuple


documents_path = '../data/pascal-sentences/ps_captions/'



documents_train = []

train_data = [f for f in listdir(documents_path) if isfile(join(documents_path, f))]


namedtupleDocument = namedtuple('namedtupleDocument', 'doc name')
for document in train_data:
    with open(documents_path+document, 'r') as f:
        doc = f.read()
        documents_train.append(namedtupleDocument(doc, document.split('.')[0]))

####
# with open(documents_path + '1.txt', 'r') as f:
#     doc = f.read()
#     new_doc = namedtupleDocument(doc, '2000')
# documents_train.append(new_doc)
####

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for document in documents_train:
    words = document[0].replace('.','').lower().split()
    tags = [document[1]]
    docs.append(analyzedDocument(words, tags))



model = gensim.models.Doc2Vec(docs, dm = 0, dbow_words= 1,size = 500, window = 8, min_count = 10, workers = 4)



# print(model.docvecs.index_to_doctag(0),'\n',model.docvecs.doctags['2000'],model.docvecs.doctags['1'])
# model.save('model_with_training_data')
# print(model.docvecs[0],'\n\n\n\n',model.docvecs[1000])

print(model.docvecs.most_similar(0))





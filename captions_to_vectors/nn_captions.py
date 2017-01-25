import tensorflow as tf
import numpy as np
import gensim

# load datasets
model = gensim.models.Doc2Vec.load('model_with_training_data')
train_data = []
test_data = []

for i, vec in enumerate(model.docvecs):
    index = model.docvecs.index_to_doctag(i)
    if index <= 400:
        train_data[index] = vec
    else:
        test_data[index-400] = vec

# load label


import gensim
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import numpy as np


np.set_printoptions(threshold=np.nan)

def docvec():
# set data path and get the list of files
    documents_path = r'../data/pascal-sentences/ps_captions/'

    train_data = [f for f in listdir(documents_path) if isfile(join(documents_path, f))]

    ## transform the data  to ndarray
    docs = [None]*len(train_data)
    for document in train_data:
        with open(documents_path+document, 'r') as f:
            doc = f.read()
            index = int(document.split('.')[0])
            docs[index-1] = doc
    docs = np.array(docs)

    # construct the tfidf value dict for each doc
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(docs)

    feature_names = vect.get_feature_names()

    docs_tfidf_dict = []
    for i in range(len(docs)):
        feature_index = tfidf[i, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf[i, x] for x in feature_index])
        dict = {}
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            dict[w] = s
        docs_tfidf_dict.append(dict)


    #### word2vec for the docs
    docs_split_org = np.array([np.array(doc.lower().replace('.','').split()) for doc in docs])

    docs_split_10 = np.array([[doc.tolist()]*10 for doc in docs_split_org])
    docs_split_10_reshape = docs_split_10.reshape(docs_split_10.shape[0]*docs_split_10.shape[1],)

    VECT_SIZE = 1000
    model = gensim.models.Word2Vec(docs_split_10_reshape,min_count=10,size=VECT_SIZE,window=5)
    # model.save('model_1')
    #
    # model = gensim.models.Word2Vec.load('model_1')


    ######## use word2vec from tensorflow
    # VECT_SIZE = 1000
    # docs_split_org = np.array([np.array(doc.lower().replace('.','').split()) for doc in docs])
    # import pickle
    #
    # final_embedding = pickle.load(open("final_embedding", "rb"))
    # dictionary = pickle.load(open("dictionary", "rb"))
    # reverse_dictionary = pickle.load(open("reverse_dictionary", "rb"))
    #
    # model = {}
    # words = [reverse_dictionary[i] for i in range(len(final_embedding))]
    #
    # for i in range(len(words)):
    #     model[words[i]] = final_embedding[i]
    ########

    # transform word2vec and doc_tfidf
    docs_vec = np.array([None] * len(docs))
    docs_tfidf = np.array([None] * len(docs))
    for i,doc in enumerate(docs_split_org):
        docs_vec[i] = np.array([None] * len(doc))
        docs_tfidf[i] = np.array([None] * len(doc))
        for j,word in enumerate(doc):
            if word in model:
                docs_vec[i][j] = np.array(model[word])
            else:
                docs_vec[i][j] = np.array([0.0]*VECT_SIZE)
            if word in docs_tfidf_dict[i]:
                docs_tfidf[i][j] = docs_tfidf_dict[i][word]
            else:
                docs_tfidf[i][j] = 0.0


    # construct TOP N doc vectors
    TOP = 10
    # docs_vec_TOP = np.zeros((len(docs),VECT_SIZE*TOP))
    docs_vec_TOP = np.zeros((len(docs),VECT_SIZE))
    for i in range(len(docs)):
        index_TOP = np.array(docs_tfidf[i]).argsort()[-TOP:][::-1]
        # vect_flat = np.zeros((1,VECT_SIZE))
        vect_flat = np.ones([VECT_SIZE])
        for index in index_TOP:
            # vect_flat = vect_flat + docs_vec[i][index]
            vect_flat = vect_flat * (docs_vec[i][index])
        docs_vec_TOP[i] = vect_flat

    return docs_vec_TOP


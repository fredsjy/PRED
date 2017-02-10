#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from skimage import io
from skimage.transform import resize
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class NormData(object):
    def __init__(self, captions_directory=None, images_directory=None, label_file=None):
        self.captions_src = captions_directory
        self.images_src = images_directory
        self.label_src = label_file
        self.captions = []
        self.images = []
        self.labels = []
        self.class_label_dict = {}
        if captions_directory is not None:
            self.caps2vec()
        if label_file is not None:
            self.labels2vec()
        if images_directory is not None:
            self.img2vec()

    def caps2vec(self):
        # extract captions and file numbers as a tuple
        file_names = [f for f in listdir(self.captions_src) if isfile(join(self.captions_src, f))]
        namedtuple_file = namedtuple('namedtuple_file', 'doc number')
        captions_names = []
        for file in file_names:
            with open(self.captions_src + file, 'r') as f:
                captions = f.read()
                captions_names.append(namedtuple_file(captions, file.split('.')[0]))

        # split file to words with number
        words_number = []
        analyzed_document = namedtuple('analyzed_document', 'words tags')
        for document in captions_names:
            words = document[0].replace('.','\n').lower().split()
            number = [document[1]]
            words_number.append(analyzed_document(words, number))
        # print(words_number[0])

        # construct a doc2vec model

        model = gensim.models.Doc2Vec(words_number, dm=0, dbow_words=1,size=1024, window=8, min_count=5, workers=4)

        # extract captions vectors and keys
        self.captions = [None] * model.docvecs.__len__()
        for i, vec in enumerate(model.docvecs):
            index = int(model.docvecs.index_to_doctag(i))
            self.captions[index-1] = vec

    def labels2vec(self):
        labels = []
        with open(self.label_src, 'r') as labels_file:
            line = labels_file.readline().rstrip()
            while line:
                labels.append(line)
                line = labels_file.readline().rstrip()

        # match label and class
        s = set(labels)
        l = list(s)
        l.sort()
        class_label_dict = {}
        i = 1
        for e in l:
            class_label_dict[e] = i
            i += 1

        self.class_label_dict = class_label_dict.copy()

        # construct labels list
        for label in labels:
            label_vec = [0] * len(s)
            label_vec[class_label_dict[label]-1] = 1
            self.labels.append(label_vec)

    def img2vec(self):
        img_names = [f for f in listdir(self.images_src) if isfile(join(self.images_src, f))]
        for img_name in img_names:
            img = io.imread(self.images_src+img_name)
            img_resize = resize(img, (28, 28))
            self.images.append(img_resize)


if __name__ == "__main__":
    # n = NormData("../data/pascal-sentences/ps_captions/", label_file="../data/pascal-sentences/labels.txt", images_directory="../data/pascal-sentences/ps_images/")
    pass

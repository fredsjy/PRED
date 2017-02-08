import pickle

import numpy as np

# ###### construct new dataset with captions and images
normData = pickle.load(open("normData","rb"))
captions_vectors = normData.captions
images_vectors = pickle.load(open('image_vec','rb'))
labels = np.array(normData.labels)

captions_images_vectors = np.array([None]*len(captions_vectors))

for i in range(len(captions_images_vectors)):
    captions_images_vectors[i] = np.append(captions_vectors[i],images_vectors[i])

##### assign
train_data = []
test_data = []
train_label = []
test_label = []
for i in range(1000):
    if i % 10 < 9:
        train_data.append(captions_images_vectors[i])
        train_label.append(labels[i])
    else:
        test_data.append(captions_images_vectors[i])
        test_label.append(labels[i])

train_data_nd = np.array(train_data)
train_label_nd = np.array(train_label)
test_data_nd = np.array(test_data)
test_label_nd = np.array(test_label)

######



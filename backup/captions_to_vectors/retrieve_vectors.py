import gensim
import numpy as np
model = gensim.models.Doc2Vec.load('model_with_training_data')
label = [None] * len(model.docvecs)
vectors = [None] * len(model.docvecs)
for i, vec in enumerate(model.docvecs):
    index = int(model.docvecs.index_to_doctag(i))
    vectors[index-1] = vec
    label[index-1] = index
vectors = np.array(vectors)

#############

print(label)

#############

<<<<<<< HEAD:captions_to_vectors/retrieve_vectors.py
# def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#   assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#   plt.figure(figsize=(18, 18))  # in inches
#   for i, label in enumerate(labels):
#     x, y = low_dim_embs[i, :]
#     plt.scatter(x, y)
#     plt.annotate(label,
#                  xy=(x, y),
#                  xytext=(5, 2),
#                  textcoords='offset points',
#                  ha='right',
#                  va='bottom')
#
#   plt.savefig(filename)
#
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# plot_only = 150
# tsne = TSNE(perplexity=30, n_components=2, n_iter=5000)
# low_dim_embs = tsne.fit_transform(X[:plot_only, :])
# labels = [label[i] for i in range(plot_only)]
# plot_with_labels(low_dim_embs, labels)

###############
=======
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plot_only = 150
tsne = TSNE(perplexity=30, n_components=2, n_iter=5000,random_state=0)
low_dim_embs = tsne.fit_transform(vectors[:plot_only, :])
labels = [label[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
>>>>>>> dev_captions_nn:backup/captions_to_vectors/retrieve_vectors.py

###############

# import numpy as np
# from matplotlib import pyplot as plt
# from tsne import bh_sne
#
# # load up data
# data = OfficialImageClassification(x_dtype="float32")
# x_data = data.all_images
# y_data = data.all_labels
#
# # convert image data to float64 matrix. float64 is need for bh_sne
# x_data = np.asarray(x_data).astype('float64')
# x_data = x_data.reshape((x_data.shape[0], -1))
#
# # For speed of computation, only run on a subset
# n = 20000
# x_data = x_data[:n]
# y_data = y_data[:n]
#
# # perform t-SNE embedding
# vis_data = bh_sne(x_data)
#
# # plot the result
# vis_x = vis_data[:, 0]
# vis_y = vis_data[:, 1]
#
# plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
# plt.colorbar(ticks=range(10))
# plt.clim(-0.5, 9.5)
# plt.show()
import pickle

final_embedding = pickle.load(open("final_embedding", "rb"))
dictionary = pickle.load(open("dictionary", "rb"))
reverse_dictionary = pickle.load(open("reverse_dictionary", "rb"))


words_vec = {}
words = [reverse_dictionary[i] for i in range(len(final_embedding))]

for i in range(len(words)):
    words_vec[words[i]] = final_embedding[i]

print(words_vec['plane'])

# low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#   labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#   plot_with_labels(low_dim_embs, labels)
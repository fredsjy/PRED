import numpy as np
import matplotlib.pyplot as plt

import pickle
from sklearn.manifold import TSNE



# load the vectors
fc2_pre = pickle.load(open('fc2_pre','rb'))
cap_vec = pickle.load(open('cap_vec','rb'))

captions_images_vectors = np.array([None]*1000)

# join the vectors
for i in range(len(captions_images_vectors)):
    captions_images_vectors[i] = np.append(cap_vec[i],fc2_pre[i])

# tsne
tsne = TSNE(perplexity=10, n_components=2, n_iter=5000, random_state=0)
low_dim_embs = tsne.fit_transform(list(captions_images_vectors[:]))

# reshape
low_dim_embs_rs = low_dim_embs.reshape((20,50,2))


# define the visualization function
colorbar = [
    '#e4007f', '#a40000','#ea68a2','#a84200','#f19149',
    '#fff45c','#8fc31f','#009944','#00736d','#0075a9',
    '#004986','#500047','#b28850','#81511c','#6a3906',
    '#59493f','#616e81','#898989','#89c997','#000000']

def visualization(data):
    fig, ax = plt.subplots(figsize=(20, 10))
    i=0
    for color in colorbar:
        x = data[i][:,0]
        y = data[i][:,1]
        scale = 20
        ax.scatter(x, y, c=color, s=scale, label="class"+str(i+1),alpha=1, edgecolors='none')
        ax.legend()
        i=i+1
    plt.show()


# visualization
visualization(low_dim_embs_rs)
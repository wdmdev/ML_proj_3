from nested_CV import run_clustering_two_layer_CV
from data_cleaner import create_dataset

from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from scipy.spatial import distance
from scipy.cluster import hierarchy
from matplotlib.pyplot import figure, title, plot, ylim, legend, show


import numpy as np
import matplotlib.pyplot as plt

data, y_true = create_dataset()

Rand = np.zeros((4,))
Jaccard = np.zeros((4,))
NMI = np.zeros((4,))
KCluster = np.zeros((4,))

linkage_list = ["complete", "average", "single", "ward"]

for i,link in enumerate(linkage_list): 
    model = AgglomerativeClustering(n_clusters= 2, affinity= 'euclidean', linkage= link)
    y_pred = model.fit_predict(data)
    Rand[i] = metrics.adjusted_rand_score(y_true,y_pred)
    Jaccard[i] = metrics.jaccard_score(y_true,y_pred)   #distance.jaccard(y_true, predicted)
    NMI[i] = metrics.normalized_mutual_info_score(y_true, y_pred,average_method= 'arithmetic')  

figure(1)
title('Cluster validity')
plot(linkage_list, Rand)
plot(linkage_list, Jaccard)
plot(linkage_list, NMI)
#legend(["Rand","Jaccard","NMI"], loc=4)

# show()

#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform(data)

# methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
title('Dendrogram ')
Z = hierarchy.linkage(data[98*3:98*4], 'single')

# hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
# fig, axes = plt.subplots(1, 2, figsize=(8, 3))
dn1 = hierarchy.dendrogram(Z, color_threshold = 6, orientation='top')
#dn2 = hierarchy.dendrogram(Z1, max_d = 4, ax=axes[1], above_threshold_color='#bcbddc', orientation='right')
#hierarchy.set_link_color_palette(None)  # reset to default after use
# print(np.array(dn1["dcoord"]).shape)
# print(len(dn1["ivl"]))

plt.show()


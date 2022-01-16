"""
    Basic code of k-means in 3-D
    and analysis in the different n-cluster
    source: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-download-auto-examples-cluster-plot-cluster-iris-py

    input:3-D array
    variable: n-cluster
    output: images of different n-clusters 
"""
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets


###############################################################
###the initial data has been preprocessing, so it is kind of hard to know the format of required data
###but for us, we can just change the input data, which should be preprocessed.

np.random.seed(5)

iris = datasets.load_iris()    ###input our data here
X = iris.data
y = iris.target

################################################################
###here comes the three n-cluster situation in the k-means clustering
###we can just change the parameter for us to use 


estimators = [                                          #this is the main difference, by changing the parameters of the KMeans()
    ("k_means_iris_8", KMeans(n_clusters=8)),           #for details, please refer to the API of KMeans from sk, it is quite clear
    ("k_means_iris_3", KMeans(n_clusters=3)),
    ("k_means_iris_bad_init", KMeans(n_clusters=3, n_init=1, init="random")),
]

fignum = 1
titles = ["8 clusters", "3 clusters", "3 clusters, bad initialization"]
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor="k")

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel("Petal width")
    ax.set_ylabel("Sepal length")
    ax.set_zlabel("Petal length")
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    if fignum == 1:
        plt.savefig("3D-8 clusters.png")
    if fignum == 2:
        plt.savefig("3D-3 clusters.png")
    if fignum == 3:
        plt.savefig("3D-bad initialization.png")
    fignum = fignum + 1

#######################################################################
###gound truth

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X[y == label, 3].mean(),
        X[y == label, 0].mean(),
        X[y == label, 2].mean() + 2,
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor="k")

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
ax.set_title("Ground Truth")
ax.dist = 12
plt.savefig("3D-gound.png")
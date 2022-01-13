# Doc Address : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
# fit(X[, y]) Fit the model with X.
# transform(X) Apply dimensionality reduction to X.
X_r = pca.fit(X).transform(X) 


"""
    printing meta-data
"""

# Percentage of variance explained for each components
print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
)

# Principal axes in feature space, representing the directions of maximum variance in the data. 
print("components are %s"%str(pca.components_)) 

# Singular values of first n_components components
print(pca.singular_values_)


"""
    end of printing meat-data
"""

plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of IRIS dataset")

plt.savefig("graph.png")
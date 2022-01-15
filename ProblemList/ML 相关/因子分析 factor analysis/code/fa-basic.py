"""
    Factor Analysis (with rotation) to visualize patterns
    Note:
    1. Applying rotations to the resulting components 
    does not inherently improve the predictive value of the derived latent space, 
    but can help visualise their structure
        * meaning this result(visualized) has been rotated

    Input: 
        X -- 2-d array X[i] means the feature array of sample i
        y -- 1-d array means the target of sample i

    Output:
        1. correlation diagram (to exibit the relations of variables)
        2. scatter diagram with respect to certain components which are 
        determined by one of three means(PCA, Unrotated FA,Varimax FA)
        3. show (by color) the contribution of each feature to first n_comps components


    link: https://scikit-learn.org/stable/auto_examples/decomposition/plot_varimax_fa.html#sphx-glr-auto-examples-decomposition-plot-varimax-fa-py
"""


import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler


from sklearn.datasets import load_iris #based on iris


######################preprocessing########################

data = load_iris()
X = StandardScaler().fit_transform(data["data"])

# print(X)
feature_names = data["feature_names"]

target_names=data.target_names

y=data.target
# print(feature_names)
#####################end_of_preprocessing#################



"""
    plot correlations of features
    X             : 2-d array X[i] means the feature array of sample i
    y             : array that lists targets
    feature_names : array(python list) that lists the feature name
    target_names  : array that lists the target names
"""

ax = plt.axes()

im = ax.imshow(np.corrcoef(X.T), cmap="RdBu_r", vmin=-1, vmax=1)

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(list(feature_names), rotation=90)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(list(feature_names))

plt.colorbar(im).ax.set_ylabel("$r$", rotation=0)
ax.set_title("Iris feature correlation matrix")
plt.tight_layout()

plt.savefig("correlation-of-fators-basic.png")

"""
    plot the target with n_comps components
"""
n_comps = 2

methods = [
    ("PCA", PCA()),
    ("Unrotated FA", FactorAnalysis()),
    ("Varimax FA", FactorAnalysis(rotation="varimax")),
]
fig, axes = plt.subplots(ncols=len(methods), figsize=(10, 8))

for ax, (method, fa) in zip(axes, methods):
    fa.set_params(n_components=n_comps)
    fa.fit(X)
    
    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2

    X_r=fa.transform(X)


    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title(method+" of IRIS dataset")

    plt.savefig(method+"_plot.png")

    """
        show the contribution of each feature to first n_comps components
    """

    components = fa.components_.T # .T means transposition of matrix
    print("\n\n %s :\n" % method)
    print(components)

    vmax = np.abs(components).max()
    ax.imshow(components, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
    ax.set_yticks(np.arange(len(feature_names)))
    if ax.is_first_col():
        ax.set_yticklabels(feature_names)
    else:
        ax.set_yticklabels([])
    ax.set_title(str(method))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Comp. 1", "Comp. 2"])

fig.suptitle("Factors")
plt.tight_layout()
plt.savefig("FA-contribution-of-feature(component-vector-by-color).png")
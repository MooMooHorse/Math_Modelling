# Code source: Andreas Mueller, Adrin Jalali
# License: BSD 3 clause


"""
    This code can do classification over any number of classes
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=27)

# y=y*10


# for i in range(0,y.shape[0]):
#     if y[i]==2 :
#         y[i]=1

# print(y)

fig, sub = plt.subplots(2, 1, figsize=(5, 8))
titles = ("break_ties = False", "break_ties = True")

for break_ties, title, ax in zip((False, True), titles, sub.flatten()):

    svm = SVC(
        kernel="linear", C=1, break_ties=break_ties, decision_function_shape="ovr"
    ).fit(X, y)

    xlim = [X[:, 0].min(), X[:, 0].max()]
    ylim = [X[:, 1].min(), X[:, 1].max()]

    xs = np.linspace(xlim[0], xlim[1], 1000)
    ys = np.linspace(ylim[0], ylim[1], 1000)
    xx, yy = np.meshgrid(xs, ys)

    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])

    colors = [plt.cm.Accent(i) for i in [0, 2, 4, 7]] # This can control the color, remember to make it large enough

    # print(colors)

    points = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Accent") # color is the same as class number, so this shall be changed and in Accent mode
    #https://matplotlib.org/stable/tutorials/colors/colormaps.html
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    #check these links to see color settings

    
    classes = [(0, 1), (0, 2), (1, 2)] # This also needs to be extended if classes number is larger than 3
    line = np.linspace(X[:, 1].min() - 5, X[:, 1].max() + 5)
    ax.imshow(
        -pred.reshape(xx.shape),
        cmap="Accent",
        alpha=0.2,
        extent=(xlim[0], xlim[1], ylim[1], ylim[0]),
    )

    for coef, intercept, col in zip(svm.coef_, svm.intercept_, classes):
        line2 = -(line * coef[1] + intercept) / coef[0]
        ax.plot(line2, line, "-", c=colors[col[0]])
        ax.plot(line2, line, "--", c=colors[col[1]])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_aspect("equal")

plt.savefig("tie.png")
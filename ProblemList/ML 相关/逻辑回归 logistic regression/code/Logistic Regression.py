"""
Logistic Regression

Do the same thing as SVM

Input & Output described below

Notes:
    1. You can see this in multinomial mode/ovr mode
    2. meshgrid(X,Y) means to remap X,Y  so that if a,b=meshgrid(X,Y) <-> a[i,j]=X[i] b[i][j]=Y[j]
    3. ravel means flatten multi-dimensional array into 1-d by simply connect the front and end of adjacent 1-d array


Link:
https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html#sphx-glr-auto-examples-linear-model-plot-logistic-multinomial-py

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression



"""
    preprocessing
"""
# make 3-class dataset for classification
centers = [[-5, 0], [0, 1.5], [5, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)
transformation = [[0.4, 0.2], [-0.4, 1.2]]
X = np.dot(X, transformation)





"""

    Input :
        X - 2-d array. X[i] shows the n feature of sample in
        y - corresponding target (class value) 


    Output:
        1. the training score, mean accuracy of training data

        2. classification result, displaying in scatter diagram


"""

for multi_class in ("multinomial", "ovr"): # You can choose to see it in either way
    clf = LogisticRegression(
        solver="sag", max_iter=100, random_state=42, multi_class=multi_class
    ).fit(X, y)

    # print the training scores
    print("training score : %.3f (%s)" % (clf.score(X, y), multi_class))

    # create a mesh to plot in
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    
    
    plt.contourf(xx, yy, Z, ) #You can add cmap=plt.cm.Paired as last parameter


    plt.title("Decision surface of LogisticRegression (%s)" % multi_class)
    plt.axis("tight")

    # Plot also the training points

    
    colors = "bry" #This needs more colors if class number >3



    for i, color in zip(clf.classes_, colors):
        # print(color)
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired, edgecolor="black", s=20
        )

    # Plot the three one-against-all classifiers
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    coef = clf.coef_
    intercept = clf.intercept_

    def plot_hyperplane(c, color):
        def line(x0):
            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

        plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)

    for i, color in zip(clf.classes_, colors):
        plot_hyperplane(i, color)

plt.savefig("lr-base.png")
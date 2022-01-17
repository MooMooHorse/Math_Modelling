# Stability and Error

 	This aspect is `crucial` for each problem-solving process.

​	The what's so-called "model name" for this is "variance analysis", but in its essence, it's not a model.

​	So I change this into `two parts` which forms an important aspect to a model, and the two parts are:

* Stability
* Error

​	This file is intended for these two parts with following supplementary notes and materials.



## Stability

Stability can be divided into two parts:

* sensitivity
* robustness



## Error

### Variance Analysis









### Cross Validation Estimation

This can be done (for machine learning aka.`ml`) with

* cross validation estimation (**for machine learning only**)
* See this link to get api
  * https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score

check the link below to see a detailed material for cross validation.

https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85

* this diagram is so vivid that I can't help transferring it to here.
  * ![img](https://miro.medium.com/max/1354/1*qPMFLEbvc8QQf38Cf77wQg.png)
* This sentence is crucial for us, so I also make it below
  * *A value of k=10 is very common in the field of applied machine learning, and is recommend if you are struggling to choose a value for your dataset.*

#### Why do we have to use cross validation(CV)?

* In this way,  we can know **how good the estimator using just training data**

* Note that we're evaluating **estimator**(pca,fa,svm,cluster,neuro network...) NOT data.

#### CV result $\rightarrow$ score

* the formulae for score can be seen in this link
  * https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators
  * click on function in the table and you can see formulae
  * it's classified by type (classification,regression,clustering)
  * **I don't know how it gets score for dimension reduction**
    * https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/
    * but according to this, pca is usually the first few steps, so you don't have to validate it.**(Most of the time, pca doesn't require machine learning)**

#### Process of CV

https://scikit-learn.org/stable/modules/cross_validation.html#a-note-on-shuffling

#### How to find the best parameters

* you can see the `score` easily not even with help of CV api, but just `clf.score()`

https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search
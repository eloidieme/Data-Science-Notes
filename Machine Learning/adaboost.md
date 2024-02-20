One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfitted. This results in new predictors focusing more and more on the hard cases. This is the technique used by AdaBoost.

A predictor is trained and tested on the training set, and then the algorithm increases the relative weight of the misclassified training instances. Then it uses those weights to train a second classifier and so on...

Once all predictors are trained, the ensemble makes predictions very much like bagging or pasting, except that predictors have different weights depending on their overall accuracy on the weighted training set.

Drawback: complicated to parallelize due to the sequential nature of the algorithm.

For the algorithm, each instance weight $w^{(i)}$ is initially set to $\frac{1}{n}$. When a predictor is trained, its error rate $r$ is computed on the training set. The error rate for the $j^{th}$ predictor can be written:
$$r_j = \frac{\sum^n_{i=1, \hat{y}_j^{(i)} \neq y^{(i)}}w^{(i)}}{\sum_{i=1}^n w^{(i)}}$$
where $\hat{y}_j^{(i)}$ is the $j^{th}$ predictor's prediction for the $i^{th}$ instance.
The predictor's weight is then computed (with $\eta$ the learning rate):
$$\alpha_j = \eta\log\frac{1 - r_j}{r_j}$$
Next, AdaBoost updates the instance weights:
$$\begin{align}&\text{for } i = 1,2,\dots,n \\ &w^{(i)} \leftarrow \begin{cases} w^{(i)} &\text{ if } \hat{y}_j^{(i)} = y^{(i)}\\ w^{(i)}\exp(\alpha_j) &\text{ if } \hat{y}_j^{(i)} \neq y^{(i)}\end{cases}\end{align}$$
Then all the instance weights are normalized.

To make predictions, AdaBoost simply computes the predictions of all the predictors and weighs them using the predictor weights:
$$\hat{y}(\mathbf{x}) = \arg \max_k \sum_{j=1, \hat{y}_j(\mathbf{x}) = k}^N \alpha_j$$
where $N$ is the total number of predictors.

Scikit-Learn uses a multiclass version of AdaBoost called SAMME (which stands for Stagewise Additive Modeling using a Multiclass Exponential loss function). When there are just two classes, SAMME is equivalent to AdaBoost. If the predictors can estimate class probabilities (i.e., if they have a predict_proba() method), Scikit-Learn can use a variant of SAMME called SAMME.R (the R stands for “Real”), which relies on class probabilities rather than predictions and generally performs better.

The classes are `AdaBoostClassifier` and `AdaBoostRegressor` in scikit-learn.
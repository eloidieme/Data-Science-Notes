Bagging is an Ensemble method where the same predictor is used but with different samples of the training set for each predictor. When sampling is performed with replacement, it's bagging (bootstrap aggregating). Otherwise, it's [[pasting]]. So bagging allow training instances to be sampled multiple times for the same predictor.

![[Pasted image 20240205185808.png]]

Then the results are aggregated (statistical mode for classification or average for regression). Generally, the ensemble has similar bias but lower variance than a single predictor trained on the original training set. These methods scale very well (each predictor can be trained in parallel).

In `scikit-learn`, we use the `BaggingClassifier` class to which we give the predictor, the number of predictors to train, the number of samples for each one and if we want bagging or pasting (with `bootstrap=True` or `False`).

Bagging is often better than pasting (more diversity so slightly higher bias but lower overall variance).

Instances that are not sampled by the `BaggingClassifier` are called out-of-bag instances (and they are different for each predictor). Since the predictor never sees those instances, it can be evaluated on them. Then the ensemble itself is evaluated by averaging out the oob evaluations of each predictor. In `scikit-learn`, we specify `oob_score = True` when creating the bagging classifier.

We can also sample features using the `BaggingClassifier` class. The methods associated with this are the [[random patches]] method and the [[random subspaces]] method.
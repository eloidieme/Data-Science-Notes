A Random Forest is an ensemble of Decision Trees, generally trained via the bagging method.

With a few exceptions, a `RandomForestClassifier` has all the hyperparameters of a `DecisionTreeClassifier` (to control how trees are grown), plus all the hyperparameters of a `BaggingClassifier` to control the ensemble itself.

The Random Forest algorithm introduces extra randomness when growing trees; instead of searching for the very best feature when splitting a node, it searches for the best feature among a random subset of features. The algorithm results in greater tree diversity, which trades a higher bias for a lower variance, generally yielding an overall better model.

It is possible to make trees even more random by also using random thresholds for each feature rather than searching for the best possible threshold. The resulting forest is an Extra-Trees ensemble. This technique trades even more bias for lower variance.

Random Forests can also compute feature importance by looking at how much the trees node that use that feature reduce Gini impurity on average (accross all trees in the forest). This is useful for feature selection.
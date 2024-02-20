Ensemble methods are methods where we aggregate the predictions of a group of predictors (the Ensemble).

-> One way of doing ensemble learning is to aggregate the predictions of each classifier and predict the class that gets the most votes. This is a hard voting classifier.
We get better results as an Ensemble than for each individual classifier. Even if each classifier is a weak learner (slightly better than random guessing), the ensemble can still be a strong learner -> this is due to the law of large numbers.
This is why Ensemble methods work best when the predictors are as independent from one another as possible.

In `scikit-learn`, we use the `VotingClassifier` class to which we give a list of classifiers and the voting method.

If all classifiers can estimate class probabilities, we can average the class probabilities over all the individual classifiers and predict the class with highest probability. This is soft voting -> often better than hard voting.

We can also do Ensemble Learning using the same predictor, with [[bagging]] and [[pasting]]. This is the method used for [[random forests]].
The learning curves are plots of the model’s performance on the training set and the validation set as a function of the training set size (or the training iteration). To generate the plots, train the model several times on different sized subsets of the training set.

![[under_learningcurve.png]]
Learning curves of an underfitting model.

When a model is underfitting, the error is quite important and doesn't decrease when the training set size increases. The error over the training set and generalization error are quite similar.

![[over_learningcurve.png]]
Learning curves of an overfitting model.

When a model is overfitting, there is a significant difference between the error over the training set and the generalization error, the latter being higher. However, the bigger the training set, the closer the two curves get.
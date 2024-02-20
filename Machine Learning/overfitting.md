Overfitting means that the model performs well on the training data, but it does not generalize well.

Complex models such as deep neural networks can detect subtle patterns in the data, but if the training set is noisy, or if it is too small (which introduces sampling noise), then the model is likely to detect patterns in the noise itself. Obviously these patterns will not generalize to new instances.

Constraining a model to make it simpler and reduce the risk of overfitting is called [[regularization]].

You need to find the right balance between fitting the training data perfectly and keeping the model simple enough to ensure that it will generalize well.
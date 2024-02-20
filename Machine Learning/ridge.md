Ridge Regression (also called Tikhonov [[regularization]]) is a regularized version of Linear Regression: a [[regularization]] term equal to $\alpha \sum_{i=1}^p \theta_i^2$ is added to the cost function.
This forces the learning algorithm to not only fit the data but also keep the model weights as small as possible. Thus, the Ridge Regression cost function is:
$$J(\mathbf{\theta}) = \text{MSE}(\mathbf{\theta}) + \alpha \frac{1}{2}\sum_{i=1}^p \theta_i^2 $$
The bias term is not regularized. The Ridge Regression closed-form solution is:
$$\hat{\mathbf{\theta}} = (X^TX + \alpha A)^{-1} X^T\mathbf{y}$$
where $A$ is the $(p+1)\times(p+1)$ identity matrix, except with a zero in the top-left cell, corresponding to the bias term.

Note that the [[regularization]] term should only be added to the cost function during training. Once the model is trained, you want to use the unregularized performance measure to evaluate the model’s performance.

The hyperparameter $\alpha$ controls how much you want to regularize the model. If $\alpha = 0$, then Ridge Regression is just Linear Regression. If $\alpha$ is very large, then all weights end up very close to zero.

If we define $\mathbf{w}$ as the vector of feature weights, then the regularization term is equal to $\frac{1}{2}||\mathbf{w}||_2^2$. For Gradient Descent, just add $\alpha \mathbf{w}$ to the MSE gradient vector.

It is important to scale the data (e.g., using a StandardScaler) before performing Ridge Regression, as it is sensitive to the scale of the input features. This is true of most regularized models.
Least Absolute Shrinkage and Selection Operator Regression (usually simply called Lasso Regression) is another regularized version of [[linear regression]]: just like [[ridge]] regression, it adds a regularization term to the cost function, but it uses the $l_1$ norm of the weight vector instead of half the square of the $l_2$ norm. 
The cost function is:
$$J(\mathbf{\theta}) = \text{MSE}(\mathbf{\theta}) + \alpha \sum_{i=1}^p |\theta_i|$$
An important characteristic of Lasso Regression is that it tends to eliminate the weights of the least important features (i.e., set them to zero).
In other words, Lasso Regression automatically performs feature selection and outputs a sparse model (i.e., with few nonzero feature weights).

The Lasso cost function is not differentiable at $\theta_i = 0 (\text{for } i = 1, 2, \dots, n)$, but Gradient Descent still works fine if you use a subgradient vector $\mathbf{g}$ instead when any $θ_i = 0$:
$$g(\mathbf{\theta},J) = \nabla_{\mathbf{\theta}}\text{MSE}(\mathbf{\theta}) + \alpha\begin{pmatrix} \text{sign} (\theta_1) \\ \text{sign} (\theta_2) \\ \vdots \\ \text{sign} (\theta_n)\end{pmatrix}$$

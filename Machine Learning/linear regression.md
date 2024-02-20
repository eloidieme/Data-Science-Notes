The goal of linear regression is to find a linear function mapping a features vector to an output. It is used to solve regression problems (i.e. where $y \in \mathbb{R}$). 
With linear regression, the hypothesis is:
$$\hat{y} = h_\theta(\mathbf{x}) = \sum_{i=0}^p \theta_i x_i$$
where $x_0=1$ and $\theta_0$ is the intercept/bias. 
In vector form, it is written as the vector product of the parameter vector $\mathbf{\theta}$ and the input vector $\mathbf{x}$:
$$\hat{y} = \mathbf{\theta}\cdot\mathbf{x} = \mathbf{\theta}^T\mathbf{x}$$
To train the linear regression model, we use the Mean Square Error function (MSE):
$$\text{MSE}(\mathbf{X}, h_{\mathbf{\theta}}) = \text{MSE}(\mathbf{\theta}) = \frac{1}{n}\sum_{i=1}^n\big( \mathbf{\theta}^T\mathbf{x}^{(i)} - y^{(i)}\big)^2$$
The cost function is then:
$$J(\theta) = \frac{1}{2}||X\theta - \mathbf{y}||_2^2 = \frac{1}{2n}\sum_{i=1}^n\big( \mathbf{\theta}^T\mathbf{x}^{(i)} - y^{(i)}\big)^2$$
So, the best parameter vector $\mathbf{\hat{\theta}}$ is the one that minimizes the mean square error:
$$\mathbf{\hat{\theta}} = \arg \min_{\theta}J(\mathbf{\theta})$$
There is a closed-form solution to this optimization problem, called the [[normal equation]]:
$$\mathbf{\hat{\theta}} = (X^TX)^{-1}X^T\mathbf{y}$$
where $X \in \mathbb{R}^{n\times p}$ is the input matrix (each row is an input vector) and $\mathbf{y} \in \mathbb{R}^{n}$ is the vector of target values. The time complexity of this method is $\mathcal{O}(p^{2.4})$ to  $\mathcal{O}(p^{3})$, depending on the implementation.

Another solution is to compute the pseudo-inverse (Moore-Penrose inverse) of $X$:
$$\mathbf{\hat{\theta}} = X^{+}\mathbf{y}$$
The pseudo-inverse is computed using SVD (Singular Value Decomposition) that can decompose $X$ into the matrix product of three matrices $U$, $\Sigma$, $V^T$. The pseudo-inverse is then computed as $X^{+} = V\Sigma^{+}U^T$. This is more efficient than computing the closed-form solution and it works when $X^TX$ is singular. This is the method `scikit-learn` uses (with the `scipy.linalg.lstsq()` function). The time complexity here is $\mathcal{O}(n^2)$. 

Alternatively, one can use [[gradient descent]] to fit a linear regressor, whether it be Batch GD, Stochastic GD or Mini-batch GD.

Comparison of algorithms for Linear Regression ($m$ is the number of instances, $n$ the number of features):

![[gd_comparison.png]]

To avoid [[overfitting]], we can apply [[regularization]] to the linear regression model. The three main methods are [[ridge]] regression, [[lasso]] regression and [[elastic net]] regression. 

It is almost always preferable to have at least a little bit of regularization, so generally plain Linear Regression is to be avoided.

Ridge is a good default, but if it is possible that only a few features are useful, Lasso or Elastic Net are to be preferred because they tend to reduce the useless features’ weights down to zero. In general, Elastic Net is preferred over Lasso because Lasso may behave erratically when the number of features is greater than the number of training instances or when several features are strongly correlated.





The normal equation is the closed-form solution to the [[linear regression]] problem.
Let $X \in \mathbb{R}^{n\times (p+1)}$ be the input or design matrix (augmented with the intercept term e.g. $1$ for each instance), $\mathbf{y}$ be the vector of target values.
Since $h_{\theta}(x^{(i)}) = (x^{(i)})^T\theta$, we have that,
$$X\theta - \mathbf{y} = \begin{bmatrix} h_{\theta}(x^{(1)}) - y^{(1)} \\ h_{\theta}(x^{(2)}) - y^{(2)} \\ \vdots \\ h_{\theta}(x^{(n)}) - y^{(n)} \end{bmatrix}.$$
And we know that for a vector $\mathbf{z}$, we have: $\mathbf{z}^T\mathbf{z} = \sum_i z_i^2 = ||\mathbf{z}||_2^2.$
With this, we can write the MSE cost function in vector form:
$$\frac{1}{2}(X\theta - \mathbf{y})^T(X\theta - \mathbf{y}) = J(\theta)$$
We then compute the gradient of the cost function $\nabla_\theta J(\theta)$ using matrix derivatives.      
$$\begin{align} \nabla_\theta J(\theta) & =\nabla_\theta\frac{1}{2}(X\theta - \mathbf{y})^T(X\theta - \mathbf{y}) \\ & = \frac{1}{2}\nabla_\theta(\theta^TX^TX\theta - \theta^TX^T - \mathbf{y}^TX\theta + \mathbf{y}^T\mathbf{y}) \quad \text{(Distributivity)} \\ &= \frac{1}{2}\nabla_\theta \text{Tr}(\theta^TX^TX\theta - \theta^TX^T - \mathbf{y}^TX\theta + \mathbf{y}^T\mathbf{y}) \quad (\forall x \in \mathbb{R}, x = \text{Tr}(x)) \\ &= 
\frac{1}{2}\nabla_\theta (\text{Tr} \ \theta^TX^TX\theta - 2\text{Tr} \ \mathbf{y}^TX\theta) \quad (\text{Tr}\ A= \text{Tr}\ A^T) \\ &= \frac{1}{2}\big( X^T X \theta + X^TX\theta - 2X^T \mathbf{y}\big) \quad (\text{Matrix derivatives}) \\ &= X^TX\theta - X^T\mathbf{y} \end{align}$$
To minimize $J$, we set its derivatives to zero, leading to the normal equations:
$$\begin{align} X^TX\hat{\theta} &= X^T\mathbf{y} \\ \Rightarrow \hat{\theta} &= (X^TX)^{-1}X^T\mathbf{y} \end{align}$$

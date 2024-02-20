Elastic Net is a middle ground between [[ridge]] regression and [[lasso]] regression. The regularization term is a simple mix of both Ridge and Lasso’s regularization terms, and you can control the mix ratio $r$. When $r = 0$, Elastic Net is equivalent to Ridge Regression, and when $r = 1$, it is equivalent to Lasso Regression.
The cost function is thus:
$$J(\mathbf{\theta}) = \text{MSE}(\mathbf{\theta}) + r\alpha\sum_{i=1}^p |\theta_i| + \frac{1-r}{2}\alpha\sum_{i=1}^n \theta_i^2$$

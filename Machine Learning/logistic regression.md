Some regression algorithms can be used for classification (and vice versa). Logistic Regression (also called Logit Regression) is commonly used to estimate the probability that an instance belongs to a particular class. If the estimated probability is greater than 50%, then the model predicts that the instance belongs to that class (called the positive class, labeled “1”), and otherwise it predicts that it does not (i.e., it belongs to the negative class, labeled “0”). This makes it a binary classifier.

Just like a [[linear regression]] model, a logistic regression model computes a weighted sum of the input features (plus a bias term), but instead of outputting the result directly like the Linear Regression model does, it outputs the logistic of this result:
$$\hat{p} = h_\mathbf{\theta}(\mathbf{x}) = \sigma(\mathbf{x}^T\mathbf{\theta})$$
The logistic—noted $\sigma(\cdot)$—is a sigmoid function (i.e., S-shaped) that outputs a number between 0 and 1.
```functionplot
---
title: Sigmoïde
xLabel: z
yLabel: a(z)
bounds: [-10,10,-0.5,1.5]
disableZoom: true
grid: false
---
a(x) = 1/(1 + exp(-x))
```
Once the Logistic Regression model has estimated the probability p = hθ(x) that an instance x belongs to the positive class, it can make its prediction ŷ easily:
$$\hat{y} = \begin{cases} 0 &\text{ if } \hat{p} < 0.5 \\ 1 &\text{ if } \hat{p} \geq 0.5\end{cases}$$
This is equivalent to saying that a Logistic Regression model predicts 1 if $t = \mathbf{x}^T\mathbf{\theta}$ is positive and 0 if it is negative. This score $t$ is often called the logit (the logit function is the inverse of the sigmoid function).
The cost function for a single instance can be written as:
$$c(\mathbf{\theta}) = \begin{cases} -\log(\hat{p}) &\text{ if } y=1 \\ -\log(1 - \hat{p}) &\text{ if } y = 0\end{cases}$$
The cost function over the whole training set is the average cost over all training instances. It can be written as a single expression called the log loss:
$$J(\mathbf{\theta}) = -\frac{1}{m}\sum_{i=1}^n \big[y^{(i)}\log\big(\hat{p}^{(i)} + (1 - y^{(i)})\log\big(1- \hat{p}^{(i)}\big)\big]$$
Just like the other linear models, Logistic Regression models can be regularized using $l_1$ or $l_2$ penalties.
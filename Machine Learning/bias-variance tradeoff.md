An important theoretical result of statistics and Machine Learning is the fact that a model’s generalization error can be expressed as the sum of three very different errors:
- Bias: This part of the generalization error is due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data.
- Variance: This part is due to the model’s excessive sensitivity to small variations in the training data. A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have high variance and thus overfit the training data.
- Irreducible error: This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data (e.g., fix the data sources, such as broken sensors, or detect and remove outliers).

![[bias_var_tradeoff.png]]
Illustration of bias and variance.

Increasing a model’s complexity will typically increase its variance and reduce its bias. Conversely, reducing a model’s complexity increases its bias and reduces its variance. This is why it is called a trade-off.


![[bias_variance_curve.png]]
Prediction error as a function of model complexity.

Mathematically, bias and variance are written:
$$\text{Bias}(\hat{f}(x))= \mathbb{E}[\hat{f}(x) - f(x)]$$
$$\text{Var}(\hat{f}(x)) = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$$
We can decompose the generalization error as follows:
$$\begin{align}
    \text{MSE}(\hat{f}(x)) &= \mathbb{E}[(y - \hat{f}(x))^2]
    \\ &= \mathbb{E}[y^2] + \mathbb{E}[\hat{f}(x)] - \mathbb{E}[2y\hat{f}]
    \\ &= \mathbb{V}[y] + \mathbb{E}[y]^2 + \mathbb{V}[\hat{f}] + \mathbb{E}[\hat{f}]^2 - 2f\mathbb{E}[\hat{f}] 
    \\ &= \mathbb{V}[y] + \mathbb{V}[\hat{f}] + (f - \mathbb{E}[\hat{f}])^2
    \\ &= \mathbb{V}[y] + \mathbb{V}[\hat{f}] + \mathbb{E}[f - \hat{f}]^2
    \\ &= \text{Bias}[\hat{f}]^2 + \mathbb{V}[\hat{f}] + \sigma^2
    \\ &=  \text{Bias}^2 + \text{Variance} + \text{Noise}
\end{align}$$







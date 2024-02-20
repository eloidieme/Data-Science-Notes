Softmax regression is a generalization of [[logistic regression]] that can support multiple classes directly, without having to train and combine multiple bineary classifiers. It is also called Multinomial Logistic Regression.

When given an instance $\mathbf{x}$, the Softmax Regression model first computes a score $s_k(\mathbf{x})$ for each class $k$, then estimates the probability of each class by applying the *softmax* function (also called the *normalized exponential*) to the scores. The equation to compute $s_k(\mathbf{x})$ is just like the equation for [[linear regression]] prediction:
$$s_k(\mathbf{x}) = \mathbf{x}^T\mathbf{\theta}^{(k)}$$
Each class has its own dedicated parameter vector $\mathbf{\theta}^{(k)}$. All these vectors are typically stored as rows in a parameter matrix $\mathbf{\Theta}$.

With the score of every class for the instance $\mathbf{x}$, we can estimate the probability $\hat{p}_k$ that the instance belongs to class $k$ by running the scores through the softmax function:
$$\hat{p}_k = \sigma(\mathbf{s}(\mathbf{x}))_k = \frac{\exp (s_k(\mathbf{x}))}{\sum_{j=1}^K \exp(s_j(\mathbf{x}))}$$
where $K$ is the number of classes.
Just like the [[logistic regression]] classifier, the Softmax Regression classifier predicts the class with the highest estimated probability:
$$\hat{y} = \arg \max_k \sigma(\mathbf{s}(\mathbf{x}))_k = \arg \max_k s_k(\mathbf{x}) = \arg \max_k \Big( \Big( \mathbf{\theta^{(k)}} \Big)^T \mathbf{x}\Big)$$
The Softmax Regression classifier predicts only one class at a time (i.e., it is multiclass, not multioutput), so it should be used only with mutually exclusive classes.

To train a softmax model, we use the cross entropy cost function, which penalizes the model when it estimates a low probability for the target class. Cross entropy is frequently used to measure how well a set of estimated class probabilities matches the target classes. The cross entropy cost function is written:
$$J(\mathbf{\Theta}) = -\frac{1}{m}\sum_{i=1}^m \sum_{k=1}^K y_k^{(i)}\log\big(\hat{p}_k^{(i)}\big)$$
$y_k^{(i)}$ is the target probability that the $i^{th}$ instance belongs to class $k$. In general, it is either equal to 1 or 0, depending on whether the instance belongs to the class or not.
When there are just two classes ($K = 2$), this cost function is equivalent to the [[logistic regression]]'s cost function.

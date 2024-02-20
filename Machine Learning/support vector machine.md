> A Support Vector Machine (SVM) is a powerful and versatile Machine Learning model, capable of performing linear or nonlinear classification, regression, and even outlier detection.

An SVM classifier not only separates two classes but also stays as far away from the closest training instances as possible. One can think of an SVM classifier as fitting the widest possible street (between the classes). This is called large margin classification.

![[svm.png]]
Large margin classification.

Notice that adding more training instances “off the street” will not affect the decision boundary at all: it is fully determined (or “supported”) by the instances located on the edge of the street. These instances are called the support vectors.

Note: SVMs are sensitive to the feature scales.

If we strictly impose that all instances must be off the street and on the right side, this is called **hard margin classification**. There are two main issues with hard margin classification. First, it only works if the data is linearly separable. Second, it is sensitive to outliers.

![[hardmargin.png]]
Hard margin sensitivity to outliers.

To avoid these issues, we use a more flexible model. The objective is to find a good balance between keeping the street as large as possible and limiting the margin violations (i.e., instances that end up in the middle of the street or even on the wrong side). This is called **soft margin classification**.

When creating an SVM model using Scikit-Learn, we can specify a number of hyperparameters. C is one of those hyperparameters. When C is small, the margin is large whereas when C is large, there are fewer margin violations.

Although linear SVM classifiers are efficient and work surprisingly well in many cases, many datasets are not even close to being linearly separable. One approach to handling nonlinear datasets is to add more features, such as polynomial features; in some cases this can result in a linearly separable dataset. That said, at a low polynomial degree, this method cannot deal with very complex datasets, and with a high polynomial degree it creates a huge number of features, making the model too slow.

When using SVMs you can apply an almost miraculous mathematical technique called the kernel trick. The kernel trick makes it possible to get the same result as if you had added many polynomial features, even with very high-degree polynomials, without actually having to add them.

Another technique to tackle nonlinear problems is to add features computed using a similarity function, which measures how much each instance resembles a particular landmark. One similarity function is the Gaussian Radial Basis Function (RBF):
$$\phi_\gamma (\mathbf{x}, l) = \exp (-\gamma||\mathbf{x} - l||^2)$$
This is a bell-shaped function varying from 0 (very far away from the landmark) to 1 (at the landmark).

The simplest approach is to create a landmark at the location of each and every instance in the dataset. Doing that creates many dimensions and thus increases the chances that the transformed training set will be linearly separable. The downside is that a training set with $n$ instances and $p$ features gets transformed into a training set with $n$ instances and $n$ features (assuming you drop the original features).

But instead of creating additional features, we can use the kernel trick with the Gaussian RBF kernel.

Other kernels exist but are used much more rarely. Some kernels are specialized for specific data structures. String kernels are sometimes used when classifying text documents or DNA sequences.

![[svmcomparison.png]]
Comparison of sklearn classes for SVM classification.

The SVM algorithm is versatile: not only does it support linear and nonlinear classification, but it also supports linear and nonlinear regression. To use SVMs for regression instead of classification, the trick is to reverse the objective: instead of trying to fit the largest possible street between two classes while limiting margin violations, SVM Regression tries to fit as many instances as possible on the street while limiting margin violations (i.e., instances off the street). The width of the street is controlled by a hyperparameter, $\epsilon$.

![[epsilonsvm.png]]
SVM Regression.

Adding more training instances within the margin does not affect the model’s predictions; thus, the model is said to be $\epsilon$-insensitive.

The linear SVM classifier model predicts the class of a new instance $\mathbf{x}$ by simply computing the decision function $\mathbf{w} \mathbf{x} + b$. If the result is positive, the predicted class $\hat{y}$ is the positive class (1), and otherwise it is the negative class (0):
$$\hat{y} = \begin{cases} 0 &\text{ if } \mathbf{w} \mathbf{x} + b < 0 \\ 1 &\text{ if } \mathbf{w} \mathbf{x} + b \geq 0\end{cases}$$
The decision boundary is the set of points where the decision function is equal to 0. The points where the decision function is equal to 1 or –1 are parallel and at equal distance to the decision boundary, and they form a margin around it. Training a linear SVM classifier means finding the values of w and b that make this margin as wide as possible while avoiding margin violations (hard margin) or limiting them (soft margin).

The slope of the decision function: it is equal to the norm of the weight vector, $||\mathbf{w}||$. In other words, dividing the slope by 2 will multiply the margin by 2.
So we want to minimize $||\mathbf{w}||$ to get a large margin. If we also want to avoid any margin violations (hard margin), then we need the decision function to be greater than 1 for all positive training instances and lower than –1 for negative training instances. If we define $t^{(i)} = –1$ for negative instances (if $y^{(i)} = 0$) and $t^{(i)} = 1$ for positive instances (if $y^{(i)} = 1$), then we can express this constraint as $t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1$ for all instances.

We can therefore express the hard margin linear SVM classifier objective as a constrained optimization problem:
$$\begin{align} \text{minimize}_{\mathbf{w},b} \quad &\frac{1}{2}\mathbf{w}^T\mathbf{w} \\ \text{subject to } \quad &t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 \text{ for } i=1,2,\dots,m \end{align}$$

To get the soft margin objective, we need to introduce a slack variable $\xi^{(i)} \geq 0$ for each instance: $\xi^{(i)}$ measures how much the $i^{th}$ instance is allowed to violate the margin. We now have two conflicting objectives: make the slack variables as small as possible to reduce the margin violations, and make $\frac{1}{2}\mathbf{w}^T\mathbf{w}$ as small as possible to increase the margin. This is where the $C$ hyperparameter comes in: it allows us to define the trade‐off between these two objectives. This gives us another constrained optimization problem:
$$\begin{align} \text{minimize}_{\mathbf{w},b} \quad &\frac{1}{2}\mathbf{w}^T\mathbf{w} + C\sum_{i=1}^n \xi^{(i)}\\ \text{subject to } \quad &t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 - \xi^{(i)} \text{ and } \xi^{(i)} \geq 0 \text{ for } i=1,2,\dots,n \end{align}$$

The hard margin and soft margin problems are both convex quadratic optimization problems with linear constraints. Such problems are known as Quadratic Programming (QP) problems. The general problem formulation is:
$$\begin{align} \text{Minimize}_{\mathbf{p}} \quad &\frac{1}{2}\mathbf{p}^T\mathbf{H}\mathbf{p} + \mathbf{f}^T\mathbf{p}\\ \text{subject to } \quad &\mathbf{A}\mathbf{p} \leq \mathbf{b} \end{align}$$
where $\mathbf{p} \in \mathbb{R}^{n_p}, \mathbf{H} \in \mathbb{R}^{n_p\times n_p}, \mathbf{f} \in \mathbb{R}^{n_p}, \mathbf{A} \in \mathbb{R}^{n_c \times n_p}, \mathbf{b} \in \mathbb{R}^{n_c}$ ($n_p$ number of parameters and $n_c$ number of constraints).

Given a constrained optimization problem, known as the primal problem, it is possible to express a different but closely related problem, called its dual problem. The solution to the dual problem typically gives a lower bound to the solution of the primal problem, but under some conditions it can have the same solution as the primal problem. Luckily, the SVM problem happens to meet these conditions, so we can choose to solve the primal problem or the dual problem; both will have the same solution. 
The dual form of the linear SVM objective is:
$$\begin{align} \text{minimize}_{\alpha} \quad &\frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha^{(i)}\alpha^{(j)} t^{(i)} t^{(j)} {\mathbf{x}^{(i)}}^T \mathbf{x}^{(j)} - \sum_{i=1}^n \alpha^{(i)} \\ \text{subject to } \quad &\alpha^{(i)} \geq 0 \text{ for } i=1,2,\dots,m \end{align}$$

And then, to get the primal solution:
$$\begin{align} \hat{\mathbf{w}} &= \sum_{i=1}^n \hat{\alpha}^{(i)}t^{(i)}\mathbf{x}^{(i)}  \\ \hat{b} &= \frac{1}{n_s}\sum_{i=1, \hat{\alpha}^{(i)} > 0} \Big( t^{(i)} - \hat{\mathbf{w}}^T\mathbf{x}^{(i)}\Big)\end{align}$$
The dual problem is faster to solve than the primal one when the number of training instances is smaller than the number of features. More importantly, the dual problem makes the [[kernel trick]] possible, while the primal does not.
Indeed, we just need to replace ${\mathbf{x}^{(i)}}^T \mathbf{x}^{(j)}$ by $K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$, with $K$ a kernel. And then, since we can't compute $\hat{\mathbf{w}}$ directly anymore to make predictions, we plug directly into the decision function to make the kernel appear:
$$h_{\hat{\mathbf{w}}, \hat{b}}(\phi(\mathbf{x}^{(n)})) = \sum_{i=1, {\hat{\alpha}}^{(i)} > 0}^n {\hat{\alpha}}^{(i)} t^{(i)} K(\mathbf{x}^{(i)}, \mathbf{x}^{(n)}) + \hat{b}$$
Here we make the dot product of the new input vector with only the support vectors. The same trick is used to compute the bias term:
$$\hat{b} = \frac{1}{n_s}\sum_{i=1,{\hat{\alpha}}^{(i)} > 0}^n\Bigg( t^{(i)} - \sum_{j=1, {\hat{\alpha}}^{(j)} > 0}^n {\hat{\alpha}}^{(j)} t^{(j)} K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) \Bigg)$$

Finally, SVMs can be used for online learning. For linear SVM classifiers, one method for implementing an online SVM classifier is to use Gradient Descent to minimize the cost function derived from the primal problem:
$$J(\mathbf{w}, b) = \frac{1}{2}\mathbf{w}^T\mathbf{w} + C\sum_{i=1}^n \max(0, 1 - t^{(i)}(\mathbf{w}^T\mathbf{x^{(i)}} + b))$$
The function $\max(0, 1 – t)$ is called the hinge loss function. It is not differentiable at $t = 1$, but just like for Lasso Regression, you can still use Gradient Descent using any subderivative at $t = 1$ (i.e., any value between –1 and 0). 
It is also possible to implement online kernelized SVMs. These kernelized SVMs are implemented in Matlab and C++. For large-scale nonlinear problems, you may want to consider using neural networks instead



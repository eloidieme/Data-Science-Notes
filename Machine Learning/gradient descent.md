Gradient Descent is a generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function. It is extensively used in [[machine learning]].

The idea is to initialize the parameters vector $\mathbf{\theta}$ with random values (*random initialization*). Then, we improve it gradually, each step attempting to decrease the cost function, until the algorithm converges to a minimum. To this end, we use one hyperparameter, the learning rate $\eta$. If the learning rate is too small, then the algorithm will have to go through many iterations to converge, which will take a long time. On the other hand, if the learning rate is too high, the algorithm will have trouble to converge and might even diverge.

![[gd_illustration.png]]
Illustration of Gradient Descent.

There are two conditions for the gradient descent algorithm to converge to the global minimum:
- the function must be convex (ensuring there is only one minimum: the global minimum).
- the derivative of the function must be Lipschitz continuous (the slope doesn't change abruptly).

For the Gradient Descent algorithm to be efficient, features must have similar scales. To ensure this, we can use [[normalization]] or [[standardization]].

![[gd_scaling.png]]
Gradient Descent with and without feature scaling.

To implement Gradient Descent, one needs to compute the gradient of the cost function with regard to each model parameter $\theta_j$. The Gradient Descent step is then:
$$\mathbf{\theta^{(\text{next step})}} = \mathbf{\theta} - \eta \nabla_{\mathbf{\theta}}J(\mathbf{\theta})$$
There are various types of Gradient Descent:
- Batch Gradient Descent: at each step, the gradient is computed using all training instances (average of gradients on all training instances).
- Stochastic Gradient Descent: at each step, one instance is randomly selected and the gradient is computed using this instance. Faster and works better with Big Data, less sensitive to local minima, but doesn't converge to optimal parameters.
- Mini-batch Gradient Descent: at each step, the gradient is computed using a subset of all training instances. A bit like batch gradient descent, but more efficient on Big Data (+ performance boost using hardware optimization of matrix operations).
On top of that, we can modify the learning rate as the algorithm goes (called Adaptive Gradient Descent): make it smaller at each step. The function that determines the learning rate at each iteration is called the learning schedule.

Usually, we let the algorithm run until the step size is within a range of $\epsilon$, a very small value.

When the cost function is convex and its slope does not change abruptly (as is the case for the MSE cost function), Batch Gradient Descent with a fixed learning rate will eventually converge to the optimal solution, but it can take O(1/$\epsilon$) iterations to reach the optimum within a range of ε, depending on the shape of the cost function. When $\epsilon$ is divided by 10, the algorithm may have to run about 10x longer.

![[sgd_mbgd_bgd.png]]
Gradient Descent paths in parameter space.

A very different way to regularize iterative learning algorithms such as Gradient Descent is to stop training as soon as the validation error reaches a minimum. This is called early stopping.
With early stopping you just stop training as soon as the validation error reaches the minimum. It is such a simple and efficient regularization technique that Geoffrey Hinton called it a “beautiful free lunch.”

![[early_stopping.png]]
Early stopping regularization.


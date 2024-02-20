In Machine Learning, a kernel is a function capable of computing the dot product $\phi(\mathbf{a})^T \phi(\mathbf{b})$, based only on the original vectors $\mathbf{a}$ and $\mathbf{b}$, without having to compute (or even to know about) the transformation $\phi$. Thus, we don't have to add features to fit nonlinear data. The most commonly used kernels are:
- Linear: $K(\mathbf{a},\mathbf{b}) = \mathbf{a}^T\mathbf{b}$
- Polynomial: $K(\mathbf{a},\mathbf{b}) = (\gamma \mathbf{a}^T \mathbf{b} + r)^d$
- Gaussian RBF: $K(\mathbf{a},\mathbf{b}) = \exp(-\gamma ||\mathbf{a} - \mathbf{b}||^2)$
- Sigmoid: $K(a,b) = \tanh (\gamma \mathbf{a}^T \mathbf{b} + r)$ 
According to Mercer’s theorem, if a function $K(\mathbf{a},\mathbf{b})$ respects a few mathematical conditions called Mercer’s conditions (e.g., $K$ must be continuous and symmetric in its arguments so that $K(\mathbf{a},\mathbf{b}) = K(\mathbf{b},\mathbf{a})$, etc.), then there exists a function $\phi$ that maps $\mathbf{a}$ and $\mathbf{b}$ into another space (possibly with much higher dimensions) such that $K(\mathbf{a},\mathbf{b}) = \phi(\mathbf{a})^T \phi(\mathbf{b})$. You can use $K$ as a kernel because you know $\phi$ exists, even if you don’t know what $\phi$ is.
Note that some frequently used kernels (such as the sigmoid kernel) don’t respect all of Mercer’s conditions, yet they generally work well in practice.


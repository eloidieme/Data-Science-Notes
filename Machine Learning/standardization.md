Standardization is a feature scaling technique using the mean and the variance to scale feature values.

First we subtract the mean value (so standardized values always have a zero mean), and then we divide by the standard deviation so that the resulting distribution has unit variance. Unlike normalization, standardization does not bound values to a specific range, which may be a problem for some algorithms (e.g., neural networks often expect an input value ranging from 0 to 1). However, standardization is much less affected by outliers.
$$\mathbf{x_{norm}} = \frac{\mathbf{x} - \mu}{\sigma}$$


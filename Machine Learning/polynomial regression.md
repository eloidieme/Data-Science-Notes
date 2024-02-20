The goal of polynomial regression is to fit a polynomial fonction mapping a features vector to an output. It is used to solve regression problems (i.e. where $y \in \mathbb{R}$). It's simply a [[linear regression]] but with added features consisting in powers of existing features. 

When there are multiple features, Polynomial Regression is capable of finding relationships between features (something a linear model cannot do). This is made possible by the fact that we also add all combinations of features up to the given degree.

Since polynomial regression models are more complex than linear regression models, then can lead to [[overfitting]]. On the other hand, linear regression models can lead to underfitting.

To determine this, we can look at the [[learning curve]]s. In case of [[overfitting]], we can use [[regularization]].
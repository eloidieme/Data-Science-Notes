A good way to reduce overfitting is to regularize the model (i.e., to constrain it): the fewer degrees of freedom it has, the harder it will be for it to overfit the data. A simple way to regularize a [[polynomial regression]] model is to reduce the number of polynomial degrees.

For a [[linear regression]] model, regularization is typically achieved by constraining the weights of the model. The three main methods are [[ridge]] regression, [[lasso]] regression and [[elastic net]] regression.

Regularization reduces the model's variance but increases its bias (i.e. [[bias-variance tradeoff]]).
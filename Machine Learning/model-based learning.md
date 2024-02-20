Machine learning models are **parameterized** with a certain number of parameters that **do not change as the size of training data changes.**

Another way to **generalize*** from a set of examples is to **build a model** of these examples and then use that **model to make predictions.**

To select the best model, you need to specify a performance measure. You can either define a:
- **Utility function (or fitness function)** that measures how good your model is.
- **Cost function** that measures how bad it is.

This performance measure is used to find parameters that best fit the training data. This is called training the model.

Model can refer to:
- type of model (e.g., Linear Regression)
- fully specified model architecture (e.g., Linear Regression with one input and one output)
- final trained model ready to be used for predictions (e.g., Linear Regression with one input and one output, using $\theta_0 = 4.85$ and $\theta_1 = 4.91 × 10–5$)

**Inference** is the stage in which a trained model is used to **infer**/predict the testing samples and comprises of a similar forward pass (parameter values) as computed in training to predict the values.

Unlike training, it doesn’t compute the error and update weights (parameter values).
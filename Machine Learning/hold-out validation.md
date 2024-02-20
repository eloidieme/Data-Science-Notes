Hold-out validation consist in holding out part of the training set to evaluate candidate models and select the best one. This set is called the validation set. More specifically, one trains multiple models with various hyperparameters on the reduced training set (i.e. the full training set minus the validation set), and one selects the model that performs best on the validation set. After this holdout validation process, one trains the best model on the full training set (including the validation set), and this gives the final model. Lastly, one evaluates this final model on the test set to get an estimate of the [[generalization]] error.

![[validation.png]]
Hold-out validation illustration.

![[validation_pipeline.png]]
Validation process.
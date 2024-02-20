Decision Trees are versatile [[machine learning]] algorithms that can perform both [[classification]] and [[regression]] tasks, and even multioutput tasks. They are powerful algorithms, capable of fitting complex datasets.

Decision Trees are also the fundamental components of [[random forests]], which are among the most powerful [[machine learning]] algorithms available today.

Practically, a Decision Tree can be visualized like this:
![[tree.png]]
The `gini` score for each node measures the impurity of the node. The lower the Gini impurity the better, so training is done by lowering the Gini impurity with each new node and finding the split that makes the largest difference. Gini impurity is calculated with this formula:
$$G_i = 1 - \sum_{k=1}^K p_{i,k}^2$$
where $p_{i,k}$ is the ratio of class $k$ instances among the training instances in the $i^{th}$ node.
For example, the depth-2 left node has a Gini score equal to $1 – (0/54)^2 – (49/54)^2 – (5/54)^2 ≈ 0.168$.

Scikit-Learn uses the Classification and Regression Tree (CART) algorithm to train Decision Trees (also called “growing” trees). The algorithm works by first splitting the training set into two subsets using a single feature $k$ and a threshold $t_k$. It chooses $k$ and $t_k$ by searching for the pair $(k, t_k)$ that produces the purest subsets (weighted by their size). The cost function is:
$$J(k, t_k) = \frac{n_{left}}{n}G_{left} + \frac{n_{right}}{n}G_{right}$$
where $n_{left/right}$ is the number of instances in the left/right subset and $G_{left/right}$ measures the Gini impurity of the left/right subset.

Decision Trees are intuitive, and their decisions are easy to interpret. Such models are often called white box models. In contrast, [[random forests]] or neural networks are generally considered black box models.

The values $p_{i,k}$ are also estimates of class probabilities for a specific node. The tree predicts the class with the largest probability.

The CART algorithm is a greedy algorithm: it greedily searches for an optimum split at the top level, then repeats the process at each subsequent level. It does not check whether or not the split will lead to the lowest possible impurity several levels down.

For Decision Trees, the overall prediction complexity is $\mathcal{O}(\log_2(n))$ independent of the number of features. The training algorithm compares all features (or less if `max_features` is set) on all samples at each node. Comparing all features on all samples at each node results in a training complexity of $\mathcal{O}(p × n \log_2(n))$. For small training sets (less than a few thousand instances), Scikit-Learn can speed up training by presorting the data (set `presort=True`), but doing that slows down training considerably for larger training sets.

By default, the Gini impurity measure is used, but you can select the entropy impurity measure instead by setting the `criterion` hyperparameter to "`entropy`".
In Machine Learning, entropy is frequently used as an impurity measure: a set’s entropy is zero when it contains instances of only one class. The definition of entropy is:
$$H_i = - \sum_{k=1, p_{i,k} \neq 0}^{K}p_{i,k}\log_2(p_{i,k})$$

Most of the time it does not make a big difference: Gini impurity and entropy lead to similar trees. Gini impurity is slightly faster to compute, so it is a good default. However, when they differ, Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees.

Decision Trees make very few assumptions about the training data (as opposed to linear models, which assume that the data is linear, for example). If left unconstrained, the tree structure will adapt itself to the training data, fitting it very closely—indeed, most likely overfitting it. Such a model is often called a nonparametric model, not because it does not have any parameters (it often has a lot) but because the number of parameters is not determined prior to training, so the model structure is free to stick closely to the data. In contrast, a parametric model, such as a linear model, has a predetermined number of parameters, so its degree of freedom is limited, reducing the risk of overfitting (but increasing the risk of underfitting).

Reducing `max_depth` will regularize the model and thus reduce the risk of overfitting. The DecisionTreeClassifier class has a few other parameters that similarly restrict the shape of the Decision Tree: `min_samples_split` (the minimum number of samples a node must have before it can be split), `min_samples_leaf` (the minimum number of samples a leaf node must have), `min_weight_fraction_leaf` (same as `min_samples_leaf` but expressed as a fraction of the total number of weighted instances), `max_leaf_nodes` (the maximum number of leaf nodes), and `max_features` (the maximum number of features that are evaluated for splitting at each node). Increasing `min_*` hyperparameters or reducing `max_*` hyperparameters will regularize the model.

Decision Trees are also capable of performing regression tasks. The main difference is that instead of predicting a class in each node, it predicts a value. The CART algorithm works mostly the same way as earlier, except that instead of trying to split the training set in a way that minimizes impurity, it now tries to split the training set in a way that minimizes the MSE. The cost function is:
$$J(k, t_k) = \frac{n_{left}}{n}\text{MSE}_{left} + \frac{n_{right}}{n}\text{MSE}_{right}$$
where $\text{MSE}_{\text{node}} = \sum_{i\in \text{node}}(\hat{y}_{\text{node}} - y^{(i)})^2$ and $\hat{y}_{\text{node}} = \frac{1}{m_{\text{node}}}\sum_{i \in \text{node}}y^{(i)}$.

Just like for classification tasks, Decision Trees are prone to overfitting when dealing with regression tasks.

Decision Trees have a few limitations: firstly, Decision Trees love orthogonal decision boundaries (all splits are perpendicular to an axis), which makes them sensitive to training set rotation. More generally, the main issue with Decision Trees is that they are very sensitive to small variations in the training data.





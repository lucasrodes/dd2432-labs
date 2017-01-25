# Brief report
Feed Forward Networks are introduced in this lab, which further explores some of their applications (e.g. *classification*, *function approximation* and *generalization*). In this regard, it first begins using only single-layer networks, where the *delta rule* is used as a training method. Later, it generalized to multilayer networks and the *generalized delta rule*.

The focus of this lab is not on the implementation but rather on the understanding of the covered topics. We note that almost all the MATLAB code is provided by the lab tutorial.

## Classification with a one layer perceptron
### Generate training data
We generate the training data using the function `sepdata.m`. In particular, it generates data matrix of dimension `d`x`N`, where `d` is the number of features and `N` is the number of samples. In our case, we use `d=2` and `N=200`. In addition it also generates the associated targets (since we examine a binary classifier, we only use two different labels).
### Implementation of the Delta rule
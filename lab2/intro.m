%% Artificial Neural Networks and other Learning Systems - Lab 2

%% 1. Introduction
%
% In this lab we use Radial Basis Functions (RBF) to approximate some simple 
% functions of one variable. Suppose we have the function $f: R 
% \to R$. RBF introduces a hidden layer such that $\hat{f}: 
% R \to R^n \to R$, where $n$ is the number of 
% neurons in the hidden layer. The trick basically consists on mapping an 
% input $x \in R$ to a new space $R^n$ using a set of 
% functions $\{\phi_i\}_{i=1}^n$ and then back to ${R}$. 
% The functions used are Gaussians with different means and, possibly, also 
% different variances
% 
% $$
% \phi_i(x) = \frac{\exp \Big(-\frac{(x-\mu_i)^2}{2\sigma_i}\Big)}{\sum_i 
% \exp \Big(-\frac{(x-\mu_i)^2}{2\sigma_i}\Big)}.
% $$
%
% _Radial_ comes from the fact that the functions $\{\phi_i\}_{i=1}^n$ 
% operate on distances rather than on the input points themselves. In this 
% regard, the selection of $\{\mu_i\}_{i=1}^n$ is essential and is typically 
% done using K-means or by simply selecting some points from the training set. 
% Given an input $x$ the output of the network is
%
% $$
% \hat{f}(x) = \sum_{i=1}^n w_i \phi_i(x).
% $$
%
% Thus we can say that the units in the hidden layer work as _basis_
% in which the function $\hat{f}$ can be expressed. 
% The motivation behind this technique is the fact that in higher 
% dimensional spaces, data is usually linearly separable. 
% Suppose we have a set of patterns $\{x_1, \dots, x_N\}$ and their 
% corresponding real function values $\{f_1, \dots, f_N\}$. While training, 
% the neural network minimizes the error measure
%
% $$
% total~error = \sum_{k=1}^N (\hat{f}_k-f_k)^2.
% $$
%
% <html><h3>Computing the weight matrix</h3></html>
%
% The weights of the network are found by solving the following system 
%
% $$\phi_1(x_1)w_1 + \phi_2(x_1)w_2 + \dots + \phi_n(x_1)w_n = f_1 $$
% 
% $$\phi_1(x_2)w_1 + \phi_2(x_2)w_2 + \dots + \phi_n(x_2)w_n = f_2 $$
%
% ...
%
% $$\phi_1(x_k)w_1 + \phi_2(x_k)w_2 + \dots + \phi_n(x_k)w_n = f_k $$
%
% ...
%
% $$\phi_1(x_N)w_1 + \phi_2(x_N)w_2 + \dots + \phi_n(x_N)w_n = f_N$$
%
% _*Question*_ What is the lower bound for the number of training examples, $N$?
% 
% From basic linear algebra, we need at least $n$ equations in a system with $n$ variables. Otherwise, the system is underdefined. Hence, we have that $N\geq n$. If the number of samples is smaller than the number of hidden units, some units will end up doing nothing or doing the same thing as other units (i.e. will activate around the same point in the input space)
% 
% _*Question*_ What happens with the error if $N=n$? Why?
% 
% If this is the case, and the system is full-rank, we have that we can find a set of weights $\{w_i\}_{i=0}^n$ such that we perfectly reconstruct the target function, i.e. $\hat{f}_k = f_k$ for all the training samples. Thus we can decrease the error to zero.
% From the network perspective what will happen is that every RBF unit will "specialize" in one single training point. For this reason, every time we feed an input there will be only one unit active and the others will be off, making it trivial for the following layer to classify the point. However, this is not a desirable behavior, because the network will have overfitted on the training points and will underperform on unseen data.
% 
% _*Question*_ Under what conditions, if any, does (\ref{eq:1}) have a solution in this case?
% 
% This happens if the rank per columns and the rank per rows is the same.
% 
% _*Question*_ During training we use an error measure defined over the training examples. Is it good to use this measure when evaluating the performance of the network? Explain!
% Although we can make the error zero on the training set, it does not mean that the network will perform well with unseen data. On the contrary, we risk to overfit the network and reduce its generalization power.
%
% <html><h3>Least Squares</h3></html>
%
% Now our error measure becomes $total~error = ||?{\Phi}{w} - \textbf{f}||^2$, which is minimized by solving $?{\Phi}^T?{\Phi}\textbf{w} = {\Phi} \textbf{f}$, i.e.
% 
% $$
% \textbf{w}_{opt} = ({\Phi}^T?{\Phi})^{-1}{\Phi}^T?\textbf{f}.
% $$
%
% <html><h3>The Delta Rule</h3></html>
%
% Sometimes, not all the sample patterns are accessible simultaneously.  If the network operates on a continuous stream of data,  the process of computing the weights changes from the previous case. We now define our error using the expectation
% 
% $$
% \xi = expected~error= \textbf{E}\Big[ 1/2 (f(x) - \hat{f}(x))^2\Big].
% $$
% 
% However, since we cannot find an exact expression for $\xi$ we use the instantaneous error as an estimate of $\xi$, i.e.
%
% $$
% \xi \approx \hat{\xi} = \frac{1}{2} (f(x) - \hat{f}(x))^2 = \frac{1}{2} e^2.
% $$
%
% Our goal here is to make $\hat{\xi} \to 0$ as fast as possible. To do so, we take a step $\Delta \textbf{w}$ in the opposite direction of the gradent of the error surface, i.e.
% 
% $$\Delta \textbf{w} = -\eta \nabla_\textbf{w} \hat{\xi} $$
%
% $$ ~= -\eta \frac{1}{2}\nabla_\textbf{w}(f(x_k)-{\Phi}(x_k)\textbf{w})^2 $$
%
% $$ ~= \eta(f(x_k)-{\Phi}(x_k)^T\textbf{w}){\Phi}(x_k) $$
%
% $$ ~= \eta e {\Phi}(x_k) $$
%
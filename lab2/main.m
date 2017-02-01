%% Artificial Neural Networks and other Learning Systems - Lab 2

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));

%% Supervised Learning of Network Weights (Batch mode training using least squares)
% Using a sinus function.
% $$f(x)=sin(2x)\ x\in[0,2\pi]$$
% * Number of samples N
% * Number of units n

% Initialization
x = (0:0.1:2*pi)';
N = size(x, 1);
f = sin(2*x);

plot_these =  [4, 5, 6, 7, 25, 56, 60];
all_residuals = [];
for units = 1:60
    makerbf;

    % Compute RBF for every input
    Phi = calcPhi(x, m, sigma2);

    % Compute least square solution for the batch
    w = Phi\f;

    % Network output
    y = Phi*w;

    if sum(units==plot_these)==1
        figure;
        rbfplot1(x, y, f, units, m, sigma2, w)
    end
    all_residuals = [all_residuals, max(abs(f-y))];
end

figure;
plot(1:60, all_residuals);
grid on;

%% Questions
% 1
% We get a residual of:
% * 1.1238 with 5 units
% * 0.1328 with 6 units
% * <0.1 with 7 units
% * <0.01 with 25 units
% * <0.001 with 56 units
%
% 2
% With 5 units we get a very high residual, with 6 units we get a
% surprisingly lower residual. Observing the plots of the RBF activations
% weighted by their weigths we notice that every time an RBF has its mean
% in correspondence to a zero in the output, its weight is pulled towards
% 0. (observe the plots with 4, 5 and 6 units)
%
% This can be explained by considering that the second has the purpose
% of "mimiking" the original function: if an RBF unit activates in proximity
% to a zero of the function, the layer must lower the corresponding weight
% to get a low output.
% 
% In the 5 unit example we can see how all 5 units end up close to the 
% zeros of $sin(2x)$ and that their associated weights are close to zero.
% This means that for every other input the output of the network will be
% close to zero and will not be able to approximate the original function.
%
% Also note how this happens every time for any number of units, at least 
% for the units placed in $0$ and $2\pi$. However if there are othe units
% that are not affected by this problem, the network performs correctly.

%% Supervised Learning of Network Weights (Batch mode training using least squares)
% Using a sinus function.
% $$f(x)=sin(2x)\ x\in[0,2\pi]$$
% * Number of samples N
% * Number of units n

% Initialization
x = (0:0.1:2*pi)';
N = size(x, 1);
f = square(2*x);

plot_these =  [4, 5, 6, 7, 25, 56, 60];
all_residuals = [];
for units = 1:100
    makerbf;

    % Compute RBF for every input
    Phi = calcPhi(x, m, sigma2);

    % Compute least square solution for the batch
    w = Phi\f;

    % Network output
    y = Phi*w;

    if sum(units==plot_these)==1
        figure;
        rbfplot1(x, y, f, units, m, sigma2, w)
    end
    all_residuals = [all_residuals, max(abs(f-y))];
end

figure;
plot(1:100, all_residuals);
grid on;

%% Questions
% 1
% We get:
% * <0.1 with 62
% * $\approx 0$ with >62
%
% 2
% It can be considered as a binary classification problem, where the
% classification to ba made could be for instance $sign(\sin (2x))$.
% 
% 3
% We've seen before that we're able to approximate $\sin (2x)$ with 6
% units. Given that this second problem can be regarded as approximating 
% $sign(\sin (2x))$, we can get a perfect result just by applying a
% threshold to the output of the previous network.
%
% We've checked it by plotting $sign(y)$ with 6 units.

%% XOR problem with RBF networks
% Consider the XOR problem in this form
%
% <latex>
% \begin{table}[]
% \centering
% \caption{My caption}
% \label{my-label}
% \begin{tabular}{ll|l}
% $x_1$ & $x_2$ & y \\ \hline
% 0     & 0     & 0 \\
% 0     & 1     & 1 \\
% 1     & 0     & 1 \\
% 1     & 1     & 0
% \end{tabular}
% \end{table}
% </latex>
%
% We can now imagine a network with 2 inputs, 4 hidden RBF units and 1
% output. We can give each of the RBF unit a mean that matches one of the
% four inputs. In this way, for every input we will have a single RBF unit
% that is highly active and the remaining will be close to zero (given that
% we choose a suitable variance). The final layer will simply need to learn
% to assign a high weight to the two units corresponding to the sencond and
% third row of the table and a low weight to the other two. We can now
% observe how we can actually get rid of the two neurons that map the
% first and last row of the table, because the sum of the activations
% performed in the last layer will be close to zero anyway. Hence the XOR
% problem can be soved by an RBF network with 2 hidden units.

%%
close all;
%% Artificial Neural Networks and other Learning Systems - Lab 2

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
addpath('info');

%% 4.1 Supervised Learning of Network Weights (Batch mode training using least squares
%
%%
% <html><h3>Using a sinus function</h3></html>
%
% Let us consider the following function $f(x)=\sin(2x)\ x\in[0,2\pi]$,
% where
%
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
title('Residual error depending on number of hidden RBF units','Interpreter', 'latex','FontSize',16);
xlabel('Number of hidden RBF units','Interpreter', 'latex','FontSize',16);
ylabel('Residual error','Interpreter', 'latex','FontSize',16)
grid on;

%%
% _*Question 1*_
% We get a residual of:
%
% * 1.1238 with 5 units
% * 0.1328 with 6 units
% * <0.1 with 7 units
% * <0.01 with 25 units
% * <0.001 with 56 units
%
% _*Question 2*_
%
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
%

%%
% <html><h3>Using a square function</h3></html>
%
% Now we consider the function $f(x)=square(2x)\ x\in[0,2\pi]$, where
%
% # Number of samples N
% # Number of units n
%

% Initialization
x = (0:0.1:2*pi)';
N = size(x, 1);
f = square(2*x);

plot_these =  [4, 5, 6, 7, 25, 56, 60];
all_residuals = [];
for units = 1:100
    %makerbf;

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
title('Residual error depending on number of hidden RBF units','Interpreter', 'latex','FontSize',16);
xlabel('Number of hidden RBF units','Interpreter', 'latex','FontSize',16);
ylabel('Residual error','Interpreter', 'latex','FontSize',16)
grid on;

%% 
% _*Question 3*_
%
% We get:
% # Less than 0.1 with more than 62 RBF units
% # $\approx 0$ with more than 63 RBF units
%
% _*Question 4*_
%
% It can be considered as a binary classification problem, where the
% classification to ba made could be for instance $sign(\sin (2x))$.
% 
% _*Question 5*_
%
% We've seen before that we're able to approximate $\sin (2x)$ with 6
% units. Given that this second problem can be regarded as approximating 
% $sign(\sin (2x))$, we can get a perfect result just by applying a
% threshold to the output of the previous network.
%
% We've checked it by plotting $sign(y)$ with 6 units.
%
% _*Question 6*_ XOR problem with RBF networks
%
% Consider the XOR problem in this form
% 
% <html>
% <table border=1><tr><td>x1</td><td>x2</td><td>y</td></tr>
% <tr><td>0</td><td>0</td><td>0</td></tr>
% <tr><td>0</td><td>1</td><td>1</td></tr>
% <tr><td>1</td><td>0</td><td>1</td></tr>
% <tr><td>1</td><td>1</td><td>0</td></tr></table>
% </html>
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

% TODO write something else about XOR

%% 4.2 On-line training using the delta rule
% _*Question 7*_
% See table for results:

clear;
rng(123);

x = (0:0.1:2*pi)';
N = size(x, 1);
fun = 'sin2x';

column_rbf = [];
column_iters = [];
column_eta = [];
column_max = [];
column_avg = [];

fun = 'sin2x';
for units = [6, 10, 40]
    for eta_pair = [[.5, .1]; [1, .8]; [2, .8]; [2,1]; [3, .8]; [4,.8]]'
        makerbf;

        totalIter = 4000;

        figure;
        iter = 0;
        itersub = 20;
        itermax = 2020;
        eta = eta_pair(1);
        color = 'r';
        diter;

        column_rbf = [column_rbf, units];
        column_iters = [column_iters, 2000];
        column_eta = [column_eta, eta];
        column_max = [column_max, max_abs];
        column_avg = [column_avg, max_mean];

        iter = 2000;
        itersub = 20;
        itermax = 2000;
        eta = eta_pair(2);
        color = 'b';
        diter;

        column_rbf = [column_rbf, units];
        column_iters = [column_iters, 2000];
        column_eta = [column_eta, eta];
        column_max = [column_max, max_abs];
        column_avg = [column_avg, max_mean];
    end;
end;

table(column_rbf', column_iters', column_eta', column_max', column_avg' ...
    , 'VariableNames', {'RBF_units', 'Iters', 'Eta', 'Final_Max', 'Final_avg'})

%%
% _*Question 8*_
% We are using a custom function:
%
% $$y = \frac{4}{5} \sin(2x+1) \log(x+1.5)$$
%

fun = 'func';
units = 10;
eta_pair = [2, 0.8];

makerbf;

totalIter = 4000;

figure;
iter = 0;
itersub = 20;
itermax = 2020;
eta = eta_pair(1);
color = 'r';
diter;

iter = 2000;
itersub = 20;
itermax = 2000;
eta = eta_pair(2);
color = 'b';
diter;

%% 
close all;
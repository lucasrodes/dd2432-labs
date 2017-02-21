%% Energy

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% Introduction
%
% In this exercise we take a deep look into the _Energy Function_. In
% particular, we are interested in it being a decreasing function as the
% states changes. Thus the dynamics must end up in an attractor. A typical
% employed energy function with this property is
% $ E = - \sum_i \sum_j w_{ij} x_i x_j$. This can be easily expressed using
% matrices and vectors as
% $E = - x^TEx$, where $x$ is a column vector with $N=1024$ coefficients.
%
%% Loading patterns and Training the weights
% 
% We load the patterns 
%
pict;

%%
% We now use the Hebbian Rule to learn the weights. We will begin by only
% using the first three patterns to train, i.e. p1, p2 and p3.rng(1);
plur = ['s',''];
patterns = [ 
    p1;
    p2;
    p3
    ];
[P, N] = size(patterns);
w = train_weights(patterns);

%% Energy at the training patterns and at their distorted counterparts
fprintf('Energy at:\n\tp1 =\t%f\n\tp11=\t%f\n', - p1 * w * p1', - p11 * w * p11')
fprintf('Energy at:\n\tp2 =\t%f\n\tp22=\t%f\n', - p2 * w * p2', - p22 * w * p22')

%%
% As one can expect, the energy at the original (training) patterns is
% lower than their distorted versions, since the training patterns are
% minima of the energy function. Nonetheless, this might not be always the
% case, since there might be other minima (attractors) of the energy
% function which might be _closer_ to the distorted patterns and with even
% lower energy than the training patterns.
%% Energy through the iterations
evolve_net(w, p11', patterns, true);
evolve_net(w, p22', patterns, true);

%% Random weight matrix (non symmetric)
% Because the weight matrix is not symmetric, this energy function does
% converge to an attractor
w = randn(N, N);
evolve_net(w, p1', -1, true);

%% Symmetric random weight matrix
% Now, the energy goes down monotonically.
w = .5 * (w + w');
evolve_net(w, p1', -1, true);

%%
close all;
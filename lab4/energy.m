%% Energy

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% Creating patterns and training
rng(1);
pict;
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

%% Energy through the iterations
evolve_net(w, p11', true);
evolve_net(w, p22', true);

%% Random weight matrix (non symmetric)
% Because the weight matrix is not symmetric, this energy function does
% converge to an attractor
w = randn(N, N);
evolve_net(w, p1', true);

%% Symmetric random weight matrix
% The energy goes down monotonically.
w = .5 * (w + w');
evolve_net(w, p1', true);
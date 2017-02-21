%% Hopfield network with 8 neurons

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
addpath('provided_code');
clc; clear; close all;

%% Introduction
%
% In this exercise we construct a Hopfield network, i.e. a fully connected
% auto-associative network with two-state neurons. In particular, we will
% be using $N = 8$ and $P = 3$ patterns.
%
% The idea in a Hopfield Network is that whenever cells are stimulated with 
% some pattern, synapses between these cells ''grow'', meaning that the
% value of the weight connecting these cells increases. Thus, the
% coefficients of the weight matrix can be written as
% $w_{ij} = 1/N \sum_{\mu=1}^P x_i^\mu x_j^\mu$
% where $\mu$ is an index within the set of patterns, $P$ is the number of
% patterns and $N$ is the number of units. This in turn can be expressed in
% matricial form as 
% $W = 1/N \sum_{\mu=1}^P x^\mu {x^\mu}^T$
% 
% In this regard, we define a function in charge of computing the weights
% 
% <include>train_weights.m</include>
%
%% Creating patterns and training (5.0)
% 
% We now define 3 patterns, and train a Hopfield Network of 8 units to
% memorize these patterns.
%

N = 8; % Number of units
P = 3; % Number of patterns
plur = ['s',''];
patterns = [ 
    vm([0 0 1 0 1 0 0 1]);
    vm([0 0 0 0 0 1 0 0]);
    vm([0 1 1 0 0 1 0 1])
    ];
% patterns = sgn(randn(P, N));
w = train_weights(patterns);

%
% We display the patterns
%

fprintf('All patterns:\n');
t0(patterns)

%% Checking that patterns are stable configurations (5.0)
% 
% We now verify that the previous three patterns are correctly stored in
% the network. To do this, we use a function in charge of updating the
% states of each neuron.
% 
% <include>evolve_net.m</include>
%
for x_in = patterns'
    [x_out, it] = evolve_net(w, x_in);
    fprintf('---- (%d iteration%s)\n', it, plur(it>1));
    fprintf('Pattern: %s\n', num2str(t0(x_in')));
    fprintf('Output : %s\n', num2str(t0(x_out')));
end

%
% We verify that all the patterns are correctly memorized by the Hopfield
% Network.
%
%% Distorted versions of the original patterns (5.1)
% 
% We now want to explore if the network is able to get to the original 
% patterns starting from distorted versions of the initial ones.
% In this regard we define three groups of distorted inputs: First one with
% two flipped bits, the second one with three flipped bits and the last one
% with four flipped bits.
%

patterns_dist = [ 
    % Patterns 1-3 with some distortion
    vm([1 0 1 0 1 0 0 1]);
    vm([1 1 0 0 0 1 0 0]);
    vm([1 1 1 0 1 1 0 1]);
    % Patterns 1-3 the first 3 bits flipped
    vm([1 1 0 0 1 0 0 1]);
    vm([1 1 1 0 0 1 0 0]);
    vm([1 0 0 0 0 1 0 1])
    % Patterns 1-3 the first 4 bits flipped
    vm([1 1 0 1 1 0 0 1]);
    vm([1 1 1 1 0 1 0 0]);
    vm([1 0 0 1 0 1 0 1])
    ];
for x_in = patterns_dist'
    [x_out, it] = evolve_net(w, x_in);
    fprintf('---- (%d iteration%s)\n', it, plur(it>1));
    fprintf('Input : %s\n', num2str(t0(x_in')));
    fprintf('Output: %s ', num2str(t0(x_out')));
    [~,indx] = ismember(x_out',patterns,'rows');
    if indx>0
        fprintf('(pattern #%d)\n', indx);
    else
        fprintf('(not a training pattern)\n');
    end
end

%%
% We observe that if the patterns have 1-3 flipped bits the network 
% reconstructs the right pattern. If the number of distorted bits is 4 or 
% more the network fails to reconstruct.
%
%% Check how many attractors are there (5.1)
%
% Continuing the previous analysis, we now want to obtain the total number 
% of attractors of the network. We know that that three of them are the
% patterns we trained the network with. In addition, we also get that some
% attractors are mixtures of these patterns. Finally, we also get the
% inverses of all these.
%
attractors = patterns;
for i = 0:2^N-1
    x_in = vm((de2bi(i, N)~=0))';
    x_out = evolve_net(w, x_in);
    if not(ismember(x_out',attractors,'rows'))
        attractors = [attractors; x_out'];
    end
end

size(attractors, 1)
t0(attractors)

%% Random confifgurations
%
% We now check if we can evolve to the original patterns from random
% configurations.

for x_in = sgn(randn(10, N))'
    [x_out, it] = evolve_net(w, x_in);
    fprintf('---- (%d iteration%s)\n', it, plur(it>1));
    fprintf('Input : %s\n', num2str(t0(x_in')));
    fprintf('Output: %s ', num2str(t0(x_out')));
    [~,indx] = ismember(x_out',patterns,'rows');
    if indx>0
        fprintf('(pattern #%d)\n', indx);
    else
        fprintf('(not a training pattern)\n');
    end
end

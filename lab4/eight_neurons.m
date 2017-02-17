%% Hopfield network with 8 neurons

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% Creating patterns and training
N = 8;
P = 3;
plur = ['s',''];
patterns = [ 
    vm([0 0 1 0 1 0 0 1]);
    vm([0 0 0 0 0 1 0 0]);
    vm([0 1 1 0 0 1 0 1])
    ];
% patterns = sgn(randn(P, N));
w = train_weights(patterns);
fprintf('All patterns:\n');
t0(patterns)

%% Checking that patterns are stable configurations
for x_in = patterns'
    [x_out, it] = evolve_net(w, x_in);
    fprintf('---- (%d iteration%s)\n', it, plur(it>1));
    fprintf('Pattern: %s\n', num2str(t0(x_in')));
    fprintf('Output : %s\n', num2str(t0(x_out')));
end

%% Checking whether random configurations evolve to the original patterns
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

%% Check if we can get to the original patterns starting from distorted versions of the initial ones
%
% If the patterns have 1-3 flipped bits the network reconstructs the right
% pattern. If the number of distorted bits is 4 or more the network fails
% to reconstruct.
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

%% Check how many attractors are there
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
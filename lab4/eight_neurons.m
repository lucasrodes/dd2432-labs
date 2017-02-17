%% Hopfield network with 8 neurons

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% Creating patterns and training
N = 8;
P = 3;
plur = ['s',''];
patterns = sgn(randn(P, N));
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
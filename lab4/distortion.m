%% Distortion Resistance

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% Creating patterns and training
pict;
patterns = [ 
    p1;
    p2;
    p3
    ];
[P, N] = size(patterns);
w = train_weights(patterns);

%% Distortion and reconstruction
% Results still depend on random initialization, but less than 20% or more
% than 80% it works most of the times.
rng(1);
for n = [10, 50, 100, 300, 400, 500, 600, 800, 900, 1010]
    figure;
    x_in = flip_img(p3, n);
    subplot(1,2,1); vis(x_in);
    title(sprintf('Flipping %d/%d = %.2f%%', n, N, n/N*100))
    [x_out, it] = evolve_net(w, x_in');
    subplot(1,2,2); vis(x_out);
    title(sprintf('Iterations: %d', it))
end
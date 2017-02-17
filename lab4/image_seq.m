%% Hopfield network with 1024 neurons

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% Creating patterns and training
pict;
plur = ['s',''];
patterns = [ 
    p1;
    p2;
    p3
    ];
[P, N] = size(patterns);

figure;
subplot(2,2,1); vis(p1); title('P1');
subplot(2,2,2); vis(p11); title('P1 degraded');
subplot(2,2,3); vis(p2); title('P2');
subplot(2,2,4); vis(p22); title('P2 mix P3');

w = train_weights(patterns);

%% Check if we can reconstruct the patterns from their degraded versions
%
% For p11, noisy version of p1, the network is able to reconstruct the
% original image in few iterations. For p22 which is a mix of p2 and p3,
% the network fails to reconstruct p2.

degraded_patterns = [
    p11;
    p22
    ];
for x_in = degraded_patterns'
    [x_out, it] = evolve_net(w, x_in);
    figure;
    subplot(1,2,1); vis(x_in);
    title('Distorted pattern')
    subplot(1,2,2); vis(x_out);
    title(sprintf('%d iteration%s', it, plur(it>1)));
end

%% Sequential update
%
% We do sequential update:
% * p11 (noisy version of p1) -> works
% * p22 (mix of p2 and p3) -> does not work with all the seeds
% * p23 (p2 with 1/4 pixels flipped) -> works
% * p24 (p2 with 1/2 pixels flipped) -> works
rng(1);
ix = randperm(N); ix = ix(1:floor(N/4)); p23 = p2; p23(1,ix) = - p23(1,ix);
ix = randperm(N); ix = ix(1:floor(N/2)); p24 = p2; p24(1,ix) = - p24(1,ix);
noise = sgn(randn(1, N));

%%
% *p11 working*
rng(1);
evolve_net(w, p11', true);
%%
% *p22 going to p3*
rng(1);
evolve_net(w, p22', true);
%%
% *p22 not working*
rng(2);
evolve_net(w, p22', true);
%%
% *p23 working*
rng(1);
evolve_net(w, p23', true);
%%
% *p24 going to p1*
rng(1);
evolve_net(w, p24', true);
%%
% *p24 not working*
rng(3);
evolve_net(w, p24', true);
%%
% *noise going to the negative of p3*
% We notice that if p3 is an attractor, also its negative must be an
% attractor because the energy function is symmetric.
rng(1);
evolve_net(w, noise', true);
%%
% *p22 going to p3 and neg p22 going to neg p3*
rng(1);
evolve_net(w, p22', true);
evolve_net(w, -p22', true);
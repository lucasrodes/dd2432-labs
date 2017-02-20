%% Hopfield network with 1024 neurons

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% Load patterns and visualization
%
% We now load the file _pict.m_ which contains nine patterns, obtained from
% nine pictures of 1024 pixels. Thus, we will use a 1024-neuron network. 
pict;

%
% Let us visualize some of the patterns from the collection. Both including
% patterns used in the training and others which are distorted versions of
% these.
figure;
subplot(2,2,1); vis(p1); title('P1', 'Interpreter', 'latex', 'Fontsize', 16);
subplot(2,2,2); vis(p11); title('P1 degraded', 'Interpreter', 'latex', 'Fontsize', 16);
subplot(2,2,3); vis(p2); title('P2', 'Interpreter', 'latex', 'Fontsize', 16);
subplot(2,2,4); vis(p22); title('P2 mix P3', 'Interpreter', 'latex', 'Fontsize', 16);
suptitle('Some patterns from the collection');

%% Training the weights
% 
% We now use the Hebbian Rule to learn the weights. We will begin by only
% using the first three patterns to train, i.e. p1, p2 and p3.
plur = ['s',''];
patterns = [ 
    p1;
    p2;
    p3
    ];
[P, N] = size(patterns);
w = train_weights(patterns);

%% Degraded patttern completion
% 
% We now inspect if the network is able to reconstruct the patterns from 
% their degraded versions
%
% From the plots, we observe that the network correclty reconstructs p1
% from p11 in few iterations. However, it is unable to find anything that
% resembles any of the three training patterns from p22.

degraded_patterns = [
    p11;
    p22
    ];
for x_in = degraded_patterns'
    [x_out, it] = evolve_net(w, x_in);
    figure;
    subplot(1,2,1); vis(x_in);
    title('Distorted pattern', 'Interpreter', 'latex', 'Fontsize', 16);
    subplot(1,2,2); vis(x_out);
    title(sprintf('Network output after %d iteration%s', it, plur(it>1))...
        , 'Interpreter', 'latex', 'Fontsize', 16);
end

%% Sequential update
%
% We do sequential update:
%
% * p11 (noisy version of p1) -> works
% * p22 (mix of p2 and p3) -> does not work with all the seeds
% * p23 (p2 with 1/4 pixels flipped) -> works
% * p24 (p2 with 1/2 pixels flipped) -> works
%
rng(1);
ix = randperm(N); ix = ix(1:floor(N/4)); p23 = p2; p23(1,ix) = - p23(1,ix);
ix = randperm(N); ix = ix(1:floor(N/2)); p24 = p2; p24(1,ix) = - p24(1,ix);
noise = sgn(randn(1, N));

%%
% *p11 working*
rng(1);
evolve_net(w, p11', patterns, true);
%%
% *p22 going to p3*
rng(1);
evolve_net(w, p22', patterns, true);
%%
% *p22 not working*
rng(2);
evolve_net(w, p22', patterns, true);
%%
% *p23 working*
rng(1);
evolve_net(w, p23', patterns, true);
%%
% *p24 going to p1*
rng(1);
evolve_net(w, p24', patterns, true);
%%
% *p24 not working*
rng(3);
evolve_net(w, p24', patterns, true);
%%
% *noise going to the negative of p3*
% We notice that if p3 is an attractor, also its negative must be an
% attractor because the energy function is symmetric.
rng(1);
evolve_net(w, noise', patterns, true);
%%
% *p22 going to p3 and neg p22 going to neg p3*
rng(1);
evolve_net(w, p22', patterns, true);
evolve_net(w, -p22', patterns, true);
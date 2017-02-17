function [ w ] = train_weights(patterns)
% patterns is a PxN matrix, where P is the number of patterns and N is the
% dimensionality of each pattern. The functions computes the NxN weight 
% matrix of a Hopfield network with N neurons.
    [P, N] = size(patterns);
    w = patterns' * patterns ./ N;
    return
end

% Tip: create 3 patterns of 8 bits like this:
% patterns = sgn(randn(3, 8))

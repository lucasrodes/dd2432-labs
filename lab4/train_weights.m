function [ w ] = train_weights(patterns, suppress_diagonal)
% patterns is a PxN matrix, where P is the number of patterns and N is the
% dimensionality of each pattern. The functions computes the NxN weight 
% matrix of a Hopfield network with N neurons.
    [P, N] = size(patterns);
    w = patterns' * patterns ./ N;
    if nargin>1 && suppress_diagonal
        w = w - diag(diag(w));
    end
end

% Tip: create 3 patterns of 8 bits like this:
% patterns = sgn(randn(3, 8))

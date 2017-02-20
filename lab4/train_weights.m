function [ w ] = train_weights(patterns, suppress_diagonal, bias)
% patterns is a PxN matrix, where P is the number of patterns and N is the
% dimensionality of each pattern. The functions computes the NxN weight 
% matrix of a Hopfield network with N neurons.
    [~, N] = size(patterns);
   
    if nargin > 2 && bias
         w = patterns' * patterns;
    elseif nargin>1 && suppress_diagonal
        w = patterns' * patterns ./ N;
        w = w - diag(diag(w));
    else
         w = patterns' * patterns ./ N;
    end
end

% Tip: create 3 patterns of 8 bits like this:
% patterns = sgn(randn(3, 8))

function [ k_neighbors ] = neighborhood2_gauss(k, sigma, grid_side) 
%NEIGHBORHOOD2 Function to get the linearized index of the neighbours as a 
% column vector input the linearized index of the winner and the threshold.
%
% * k: unit to find neighbours of
% * sigma: std of the gaussian used to weight the unit's updates
%          E.g. if sigma = 1, then at distance 3 the weight is 0.011
% * grid_size: size of the grid, in the lab example it is 10x10
%
    [x, y] = meshgrid(1:grid_side, 1:grid_side);
    is = reshape(x, 1, grid_side^2);
    js = reshape(y, 1, grid_side^2);
    i = is(k);
    j = js(k);
    dist = abs(x-i) + abs(y-j);
    k_neighbors = reshape(exp(-.5*dist.^2/sigma^2), grid_side^2, 1);
end
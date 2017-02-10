function [ k_neighbors, distances ] = neighborhood2(k, radius, grid_side) 
%NEIGHBORHOOD2 Function to get the linearized index of the neighbours, given the linearized index of the winner and the threshold.
    [x, y] = meshgrid(1:grid_side, 1:grid_side);
    is = reshape(x, 1, grid_side^2);
    js = reshape(y, 1, grid_side^2);
    i = is(k);
    j = js(k);
    dist = abs(x-i) + abs(y-j);
    k_neighbors = reshape(dist - radius, 1, grid_side^2) <= 0;
end
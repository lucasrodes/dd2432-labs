%% Cyclic Tour

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% Introduction
% For a circular structure we to define a different neighborhood function.
% We can just pad the array of the units with additional units on the
% sides:
%
% $$[9,10|1,2,3,4,5,6,\underbrace{7,8,\mathbf{9},10|1}_{neighborhood},2]$$
%

%% Clustering with 10 units
% We will try many options:
% * Change the number of units (more than 10 will result in better paths, 
% at the cost that some units will overlap or will not be near a city)
% * Randomizing the order in which the cities are considered at every epoch
% * Change the number of epochs (we note that most of the time the 
% cycle oscillates periodically between two configurations, so it's better 
% to switch to a different neighborhood size more fastly)
% * Change the threshold values that decide the size of the neighborhood
% based on the epoch number
%
% *Setup*

cities;
num_of_epochs = 200;
eta = 0.2;
[num_of_cities, num_of_features] = size(city); 
num_of_units = 20;
weights = rand(num_of_units, num_of_features);

max_neighborhood_size = 3;
padded = [num_of_units - max_neighborhood_size : num_of_units, ...
            1 : num_of_units, ...
            1 : num_of_units + max_neighborhood_size];

%%
% *Training*

figure;
for epoch = 1:num_of_epochs
    if epoch < .2 * num_of_epochs
        neighborhood_size = 3;
    elseif epoch < .5 * num_of_epochs
        neighborhood_size = 2;
    elseif epoch < .8 * num_of_epochs
        neighborhood_size = 1;
    else
        neighborhood_size = 0;
    end

    for city_idx = randperm(num_of_cities)
        % Find winner unit
        city_coor = city(city_idx,:);
        diff = repmat(city_coor, num_of_units, 1) - weights;
        [~, winning_unit] = min(sum(diff.^2, 2));
        
        % Neighbors
        neighbors_idx = padded(winning_unit+max_neighborhood_size+1-neighborhood_size:winning_unit+max_neighborhood_size+1+neighborhood_size);
        
        % Update mask
        update_function = zeros(num_of_units, num_of_features);
        update_function(neighbors_idx, :) = 1;
        
        % Update weights
        weights = weights + update_function .* (eta * diff);
        
        % Plotting
        tour = [weights; weights(1,:)];
        plot(tour(:,1),tour(:,2),'c-o', 'LineWidth',1); hold on;
        plot(weights(winning_unit,1),weights(winning_unit,2),'.b','MarkerSize', 35);
        plot(weights(neighbors_idx,1),weights(neighbors_idx,2),'.g-','MarkerSize', 20, 'LineWidth',2);
        plot(city(:,1),city(:,2),'.m','MarkerSize', 15);
        plot(city(city_idx,1),city(city_idx,2),'.r','MarkerSize', 35);
        title(sprintf('Epoch: %4d, Neighborhood size: %2d', epoch, neighborhood_size));
        hold off;
        pause(0.0001);
    end
end

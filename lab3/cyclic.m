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

%% Experiment
%
% In this experiment we are supposed to use SOM to find an approximation to 
% the TSP problem. In this regard, given a set of $N$ points (cities) and 
% an array of topologycally-arranged units (circular-fashion) we use SOM in 
% order to find a cycle that goes through each city. Ideally, we want to 
% use $N$ units such that each of them clusters a specific city and hence a
% closed cycle is formed.
%
% However this is not typically the case and some units might end up
% not clustering any of the given points. 
%
% For this experiment we need to set different parameters, which as
% proposed in the lab tutorial are set using _trial and error_ methodology.
% This parameters are:
%
% * *Number of units* (we are told to use 10 units, however we explored other
% configurations with more than 10 units which actually lead to "better"
% results.
% * *Number of epochs*. We need enough iterations such that we can obtain a
% reasonable result. However, at a certain point the result remain
% unvariant. In particular, we note that most of the time the cycle 
% oscillates periodically between two configurations, so it's better 
% to switch to a different neighborhood size faster).
% * *Learning rate $\eta$*. We use a constant learning rate, however it
% might be interesting to decrease it as the number of iterations
% increases.
% * *Neighbourhood size*. We start with a biger neighbourhood and decrease
% it as the number of iterations increases.
%
% Besides this, we have to pay attention to the following consideration:
%
% * Randomization of the order in which the cities are considered at every 
% epoch (avoid possible correlations).
%
%
% *Setup*

cities;
num_of_epochs = 200;
eta = 0.2;
[num_of_cities, num_of_features] = size(city); 

%% 
% *Training with 10 units*
%
% Let us first consider the case where we use the same number of units as
% the number of cities.

num_of_units = 10;
weights = rand(num_of_units, num_of_features);

max_neighborhood_size = 3;
padded = [num_of_units - max_neighborhood_size : num_of_units, ...
            1 : num_of_units, ...
            1 : num_of_units + max_neighborhood_size];
        
figure(1);
filename = './html/cycle_10units.gif';

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
        neighbors_idx = padded(winning_unit+max_neighborhood_size+...
            1-neighborhood_size:winning_unit+max_neighborhood_size+...
            1+neighborhood_size);

        % Update mask
        update_function = zeros(num_of_units, num_of_features);
        update_function(neighbors_idx, :) = 1;

        % Update weights
        weights = weights + update_function .* (eta * diff);

        % Plotting
        tour = [weights; weights(1,:)];
        plot(tour(:,1),tour(:,2),'c-o', 'LineWidth',1); hold on;
        plot(weights(winning_unit,1),weights(winning_unit,2),'.b',...
            'MarkerSize', 35);
        plot(weights(neighbors_idx,1),weights(neighbors_idx,2),'.g-',...
            'MarkerSize', 20, 'LineWidth',2);
        plot(city(:,1),city(:,2),'.m','MarkerSize', 15);
        plot(city(city_idx,1),city(city_idx,2),'.r','MarkerSize', 35);
        title(sprintf(...
            'Number of units: %4d, Epoch: %4d, Neighborhood size: %2d', ...
            num_of_units, epoch,neighborhood_size), 'Interpreter', ...
            'latex', 'Fontsize', 16);
        xlabel('$x$', 'Interpreter', 'latex', 'Fontsize', 13);
        ylabel('$y$', 'Interpreter', 'latex', 'Fontsize', 13);  
        hold off;
        
        drawnow
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if epoch == 1;
            imwrite(imind,cm,filename,'gif','DelayTime',0, 'Loopcount',inf);
        else
            imwrite(imind,cm,filename,'gif','DelayTime',0,'WriteMode','append');
        end
        %pause(0.0001);
    end
end

%%
%
% <<cycle_10units.gif>>
% 

%% 
% *Training with 20 units*
%
% The results obtained with 10 units are close to the optimal solution,
% however, some of the units do not cluster any point leaving the cycle
% incomplete (we do not visit some cities). In this regard, we now increase
% the number of units, such that we can ensure that _all_ cities are
% visited. Once finished, we can obtain the order of visited cities by
% only looking at the units that cluster a given point.
%

num_of_units = 20;
weights = rand(num_of_units, num_of_features);

max_neighborhood_size = 3;
padded = [num_of_units - max_neighborhood_size : num_of_units, ...
            1 : num_of_units, ...
            1 : num_of_units + max_neighborhood_size];
        
figure(2);
filename = './html/cycle_20units.gif';
f = gcf;

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
        neighbors_idx = padded(winning_unit+max_neighborhood_size+...
            1-neighborhood_size:winning_unit+max_neighborhood_size+...
            1+neighborhood_size);

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
        title(sprintf(...
            'Number of units: %4d, Epoch: %4d, Neighborhood size: %2d', ...
            num_of_units, epoch,neighborhood_size), 'Interpreter', ...
            'latex', 'Fontsize', 16);
        xlabel('$x$', 'Interpreter', 'latex', 'Fontsize', 13);
        ylabel('$y$', 'Interpreter', 'latex', 'Fontsize', 13);
        hold off;
        drawnow
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if epoch == 1;
            imwrite(imind,cm,filename,'gif','DelayTime',0, 'Loopcount',inf);
        else
            imwrite(imind,cm,filename,'gif','DelayTime',0,'WriteMode','append');
        end
    end
end

%%
%
% <<cycle_20units.gif>>
% 

%% Conclusions
%
%%
% 
% * We could define a convergence criteria, in order to stop the iterations
% once a solution is found (i.e. the cycle does not change).
% * Number of units should be greater than the number of points.
% 


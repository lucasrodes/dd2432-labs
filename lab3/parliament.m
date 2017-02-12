%% Data Clustering: Votes of MPs

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
warning('off', 'MATLAB:mode:EmptyInput')
clc; clear; close all;
addpath('provided_code');
%% Introduction
%
% In this exercise we will try to get some insights of the swedish parlia-
% ment ''political distribution'' by exploring data corresponding to the
% different MPs (such as their votes, their origins etc.). 
%
% In this regard, we will work with variables loaded from |politics|:
%
% * |parties| = party membership of each MP
% * |sex| = sex of each MP
% * |districts| = district of each MP
% * |votes| = list of 31 votes for each MP
% * |names| = name of each MP
%
% Plus some additional variables:
%
% * |sex_colormap| = colormap for sex
% * |sex_labels| = labels for sex
% * |party_colormap| = colormap for party
% * |party_labels| = labels for party
% * |districts_colormap| = colormap for district
%
% For instance, let us display some of these labels for the 10 first MPs.
politics;
table(names(1:10), sex_labels(((sex(1:10) + 1)')), ...
    party_labels(((parties(1:10) + 1)')), districts(1:10), ...
    'VariableNames', {'Name', 'Sex', 'Party', 'District'})

%% Setup and Topology
% *Setup*
eta = 0.2;
num_of_epochs = 1000;
[num_of_MP, num_of_votes] = size(votes);

%%
% *Topology*
%
% We will work with a 2D topology in which the units are connected in a
% grid-like fashion. 
%
% From an abstract point of view the units are organized in a 2D square 
% grid, and their coordinates are given by a $(i, j)$ pair. In practice, we
% will assign an index $k$ to every unit so that $k = side \cdot i + j$, 
% where $side = 10$ in this example.

side_of_topologic_grid = 10;
num_of_units = side_of_topologic_grid^2;
[x, y] = meshgrid(1:side_of_topologic_grid, 1:side_of_topologic_grid);
is = reshape(x, 1, num_of_units);
js = reshape(y, 1, num_of_units);
weights = rand(num_of_units, num_of_votes);

%%
% *Neighbourhood function*
%
% Self Organizing Maps differ from Competitive Leaning in that not only the
% winning unit is updated but also its neighbours. This might look alike
% the shared learning in Competitive Learning but differs in the way the
% neighbourhood is defined. We now work with a new space in the output
% layer where the units are arranged acording to a particular topology that
% we choose. As we highlighted before, usually simple topologies are chosen
% (in our example we use 2D grid). 
%
% In this regard, we need to define what the neighbourhood of a unit is.
%
%%
%
% _Manhattan distance_
%
% A first attempt is to define the neighborhood of a unit by thresholding 
% the manhattan distance between the unit and the rest of units to a
% certain value we call radius.This is actually the way that is proposed in
% the lab tutorial. We then update the winning unit and its neighbours' 
% centers using the same function, i.e.
% $ w_k^{new} \gets w_k^{old} -\eta (x-w_k)$, where $x$ is the input sample
% activating the wining unit and $k$ is the winning unit's and its 
% neighbours unit's indices.
%
% <include>neighborhood2.m</include>
%

%%
figure;

%
% *Example 1*
% Let us consider a first example with the grid $10 \times 10$, where 
% the winner is, say, the $27:th$ unit and the neighbourhood is of radius 
% 2.
%

k_winner = 27;
radius = 2;
k_neighbors = neighborhood2(k_winner, radius, side_of_topologic_grid);

subplot(1,2,1);
hold on;
plot(js, is, '.b', 'MarkerSize', 15);
plot(js(k_neighbors), is(k_neighbors), '.m', 'MarkerSize', 30);
plot(js(k_winner), is(k_winner), '.r', 'MarkerSize', 40);
title(sprintf('Winner: (%d,%d)    Radius: %d', ...
   is(k_winner), js(k_winner), radius), 'Interpreter', 'latex', ...
   'FontSize', 16),
axis ij;
axis image;
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);


%
% *Example 2*
% In this example the winning unit is 67 and we consider a larger
% neighbourhood, namely of radius 4.
%

k_winner = 67;
radius = 4;
k_neighbors = neighborhood2(k_winner, radius, side_of_topologic_grid);

subplot(1,2,2);
hold on;
plot(js, is, '.b', 'MarkerSize', 15);
plot(js(k_neighbors), is(k_neighbors), '.m', 'MarkerSize', 30);
plot(js(k_winner), is(k_winner), '.r', 'MarkerSize', 40);
title(sprintf('Winner: (%d,%d)    Radius: %d', ...
    is(k_winner), js(k_winner), radius), 'Interpreter', 'latex', ...
    'FontSize', 16),
axis ij;
axis image;
xlim([0, side_of_topologic_grid+1]);
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
ylim([0, side_of_topologic_grid+1]);


suptitle('Example using Manhattan Distance neighbourhood');
%%
%
%
% _Gaussian neighbourhood level_
%
% <include>neighborhood2_gauss.m</include>
%
% In the following, we propose to introduce a slight modification on the 
% neighbourhood obtention. So far we have assigned the same update effect
% (unitary) to all updated units, both the winner and its neighbours. We 
% now consider the usage of a gaussian distribution centered at the winner 
% unit, which tells which units are more affected by the seen point. In
% particular, the update equation is now given by $w_k^{new} = w_k^{old} -\eta h_k (x-w)$, where $h_k$ denotes the 
% _neighborhood_ level if unit $k$ which decreases as the distance to the 
% winning unit increases.

%%
figure;

%
% *Example 1*
%
k_winner = 27;
sigma = 1;
k_neighbors = neighborhood2_gauss(k_winner, sigma, side_of_topologic_grid);

subplot(1,2,1);
b = bar3(reshape(k_neighbors,side_of_topologic_grid,side_of_topologic_grid));
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end

title(sprintf('Winner: (%d,%d)    Sigma: %.2f', ...
    is(k_winner), js(k_winner), sigma), 'Interpreter', 'latex', ...
   'FontSize', 16),
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);

%
% *Example 2*
%
k_winner = 27;
sigma = 0.5;
k_neighbors = neighborhood2_gauss(k_winner, sigma, side_of_topologic_grid);

subplot(1,2,2);
b = bar3(reshape(k_neighbors, side_of_topologic_grid, side_of_topologic_grid));
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end

title(sprintf('Winner: (%d,%d)    Sigma: %.2f', ...
    is(k_winner), js(k_winner), sigma), 'Interpreter', 'latex', ...
   'FontSize', 16),
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);


colorbar
suptitle('Example using Neighbourhood level');

%% Training - Manhattan Distance
%
% As usual, we diminish the size of the neighborhood as we go through the
% training epochs. Also we shuffle the MPs at every epoch before presenting
% them sequentially to the network;

for epoch = 1:num_of_epochs
    if epoch < .1 * num_of_epochs
        radius = 4;
        sigma = 2;
    elseif epoch < .2 * num_of_epochs
        radius = 3;
        sigma = 1.6;
    elseif epoch < .5 * num_of_epochs
        radius = 2;
        sigma = 1.2;
    elseif epoch < .8 * num_of_epochs
        radius = 1;
        sigma = 0.8;
    else
        radius = 0;
        sigma = 0.00001;
    end
    for mp_idx = randperm(num_of_MP)
        % Find winning unit
        mp = votes(mp_idx, :);
        diff = repmat(mp, num_of_units, 1) - weights;
        dist = sum(diff.^2, 2);
        [~, k_winner] = min(dist);
        
        % Update function (is a col. vector) and update mask (just repeat
        % it for num_of_votes columns)
        update_function = repmat( ...
            neighborhood2(k_winner, radius, side_of_topologic_grid), ...
            1, num_of_votes);
        %update_function = repmat( ...
        %    neighborhood2_gauss(k_winner, sigma, side_of_topologic_grid), ...
        %    1, num_of_votes);
        
        % Update weights
        weights = weights + update_function .* (eta * diff);
    end
end
%% Visualizing - Manhattan Distance
% *Explanation for the unit distribution*
%
% * For every MP, find the closest unit. In short, cluster the MPs using
% the grid of units.
% * Then assign to every unit a set of colors depending on the attribute
% distribution within the corresponding cluster.
% * Plot the units in the topological space with their color for each 
% attribute.

clustering = zeros(num_of_MP, 1);
sex_freq = cell(num_of_units, 1);
party_freq = cell(num_of_units, 1);
districts_freq = cell(num_of_units, 1);
for mp_idx = 1:num_of_MP
    % Obtain the closest unit to mp
    mp = votes(mp_idx, :);
    diff = repmat(mp, num_of_units, 1) - weights;
    dist = sum(diff.^2, 2);
    [~, k_winning_unit] = min(dist);
    
    % Assign mp attributes to the found unit
    clustering(mp_idx) = k_winning_unit;
    sex_freq{k_winning_unit} = [sex_freq{k_winning_unit}, sex(mp_idx)];
    party_freq{k_winning_unit} = [party_freq{k_winning_unit}, parties(mp_idx)];
    districts_freq{k_winning_unit} = [districts_freq{k_winning_unit}, districts(mp_idx)];
end

% Plot grid of units
figure;
for k = 1:num_of_units
    
    % Plot sex attributes
    subplot(1,3,1); hold on;
    if(~isempty(sex_freq{k}))
        for l = unique(sort(sex_freq{k}), 'stable')
            marker_size = nnz(sex_freq{k}==l)^1.2*100;
            scatter(js(k), is(k), marker_size, 'filled', ...
                'MarkerFaceColor', sex_colormap(l+1,:), ...
                'MarkerFaceAlpha',4/8);  
        end
    end
    
    % Plot party attributes
    subplot(1,3,2); hold on;
    if(~isempty(party_freq{k}))
        for l = unique(sort(party_freq{k}), 'stable')
            marker_size = (nnz(party_freq{k}==l))^1.2*100;
            scatter(js(k), is(k), marker_size, 'filled', ...
            'MarkerFaceColor', party_colormap(l+1,:), ...
            'MarkerFaceAlpha',4/8);
        end
    end
    
    % Plot district attributes
    subplot(1,3,3); hold on;
    if(~isempty(districts_freq{k}))
        for l = unique(sort(districts_freq{k}), 'stable')
            marker_size = nnz(districts_freq{k}==l)^1.2*100;
            scatter(js(k), is(k), marker_size, 'filled', ...
            'MarkerFaceColor', districts_colormap(l,:), ...
            'MarkerFaceAlpha',4/8);
        end
    end
end

subplot(1,3,1);
title('Sex per unit', 'Interpreter', 'latex', 'FontSize', 16);
axis ij;
axis image;
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);

h = zeros(length(sex_labels), 1);
for i = 1:length(sex_labels)
    h(i) = plot(NaN, NaN, 'o','color', sex_colormap(i, :)); hold on;
end
legend(h, sex_labels,'Location', 'southoutside', 'Interpreter', ...
    'latex', 'FontSize', 13);


subplot(1,3,2);
title('Party per unit', 'Interpreter', 'latex', 'FontSize', 16);
axis ij;
axis image;
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);

h = zeros(length(party_names), 1);
for i = 1:length(party_names)
    h(i) = plot(NaN, NaN, 'o','color', party_colormap(i, :)); hold on;
end
legend(h, party_names,'Location', 'southoutside', 'Interpreter', ...
    'latex', 'FontSize', 13);

subplot(1,3,3);
title('District per unit', 'Interpreter', 'latex', 'FontSize', 16);
axis ij;
axis image;
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);


suptitle('SOM for different attributes, using Manhattan Distance neighbourhood');

%%
% * Tables of clusters *
%

%%
% _Association of every MP to one unit_
[~, order] = sort(clustering);
t_manhattan = table(names(order), sex_labels(((sex(order) + 1)')), ...
    party_labels(((parties(order) + 1)')), ...
    districts(order), ...
    clustering(order), ...
    'VariableNames',{'Name','Sex','Party','District','Cluster'})

%%
% _Clusters with 4 ore more MPs_
for cl = unique(clustering)'
    if (sum(clustering(order) == cl) > 3)
        t_manhattan(clustering(order) == cl, 1:5);
    end
end

%% Training - Gaussian Neighbourhood Level
%
% As usual, we diminish the size of the neighborhood as we go through the
% training epochs. Also we shuffle the MPs at every epoch before presenting
% them sequentially to the network;

for epoch = 1:num_of_epochs
    if epoch < .1 * num_of_epochs
        radius = 4;
        sigma = 2;
    elseif epoch < .2 * num_of_epochs
        radius = 3;
        sigma = 1.6;
    elseif epoch < .5 * num_of_epochs
        radius = 2;
        sigma = 1.2;
    elseif epoch < .8 * num_of_epochs
        radius = 1;
        sigma = 0.8;
    else
        radius = 0;
        sigma = 0.00001;
    end
    for mp_idx = randperm(num_of_MP)
        % Find winning unit
        mp = votes(mp_idx, :);
        diff = repmat(mp, num_of_units, 1) - weights;
        dist = sum(diff.^2, 2);
        [~, k_winner] = min(dist);
        
        % Update function (is a col. vector) and update mask (just repeat
        % it for num_of_votes columns)
        update_function = repmat( ...
            neighborhood2(k_winner, radius, side_of_topologic_grid), ...
            1, num_of_votes);
        %update_function = repmat( ...
        %    neighborhood2_gauss(k_winner, sigma, side_of_topologic_grid), ...
        %    1, num_of_votes);
        
        % Update weights
        weights = weights + update_function .* (eta * diff);
    end
end
%% Visualizing - Gaussian Neighbourhood Level
% *Explanation for the unit distribution*
%
% * For every MP, find the closest unit. In short, cluster the MPs using
% the grid of units.
% * Then assign to every unit a set of colors depending on the attribute
% distribution within the corresponding cluster.
% * Plot the units in the topological space with their color for each 
% attribute.

clustering = zeros(num_of_MP, 1);
sex_freq = cell(num_of_units, 1);
party_freq = cell(num_of_units, 1);
districts_freq = cell(num_of_units, 1);
for mp_idx = 1:num_of_MP
    % Obtain the closest unit to mp
    mp = votes(mp_idx, :);
    diff = repmat(mp, num_of_units, 1) - weights;
    dist = sum(diff.^2, 2);
    [~, k_winning_unit] = min(dist);
    
    % Assign mp attributes to the found unit
    clustering(mp_idx) = k_winning_unit;
    sex_freq{k_winning_unit} = [sex_freq{k_winning_unit}, sex(mp_idx)];
    party_freq{k_winning_unit} = [party_freq{k_winning_unit}, parties(mp_idx)];
    districts_freq{k_winning_unit} = [districts_freq{k_winning_unit}, districts(mp_idx)];
end

% Plot grid of units
figure;
for k = 1:num_of_units
    
    % Plot sex attributes
    subplot(1,3,1); hold on;
    if(~isempty(sex_freq{k}))
        for l = unique(sort(sex_freq{k}), 'stable')
            marker_size = nnz(sex_freq{k}==l)^1.2*100;
            scatter(js(k), is(k), marker_size, 'filled', ...
                'MarkerFaceColor', sex_colormap(l+1,:), ...
                'MarkerFaceAlpha',4/8);  
        end
    end
    
    % Plot party attributes
    subplot(1,3,2); hold on;
    if(~isempty(party_freq{k}))
        for l = unique(sort(party_freq{k}), 'stable')
            marker_size = (nnz(party_freq{k}==l))^1.2*100;
            scatter(js(k), is(k), marker_size, 'filled', ...
            'MarkerFaceColor', party_colormap(l+1,:), ...
            'MarkerFaceAlpha',4/8);
        end
    end
    
    % Plot district attributes
    subplot(1,3,3); hold on;
    if(~isempty(districts_freq{k}))
        for l = unique(sort(districts_freq{k}), 'stable')
            marker_size = nnz(districts_freq{k}==l)^1.2*100;
            scatter(js(k), is(k), marker_size, 'filled', ...
            'MarkerFaceColor', districts_colormap(l,:), ...
            'MarkerFaceAlpha',4/8);
        end
    end
end

subplot(1,3,1);
title('Sex per unit', 'Interpreter', 'latex', 'FontSize', 16);
axis ij;
axis image;
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);

h = zeros(length(sex_labels), 1);
for i = 1:length(sex_labels)
    h(i) = plot(NaN, NaN, 'o','color', sex_colormap(i, :)); hold on;
end
legend(h, sex_labels,'Location', 'southoutside', 'Interpreter', ...
    'latex', 'FontSize', 13);


subplot(1,3,2);
title('Party per unit', 'Interpreter', 'latex', 'FontSize', 16);
axis ij;
axis image;
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);

h = zeros(length(party_names), 1);
for i = 1:length(party_names)
    h(i) = plot(NaN, NaN, 'o','color', party_colormap(i, :)); hold on;
end
legend(h, party_names,'Location', 'southoutside', 'Interpreter', ...
    'latex', 'FontSize', 13);

subplot(1,3,3);
title('District per unit', 'Interpreter', 'latex', 'FontSize', 16);
axis ij;
axis image;
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);


suptitle('SOM for different attributes, using Gaussian Neighbourhood Level');

%%
% * Tables of clusters *
%

%%
% _Association of every MP to one unit_
[~, order] = sort(clustering);
t_manhattan = table(names(order), sex_labels(((sex(order) + 1)')), ...
    party_labels(((parties(order) + 1)')), ...
    districts(order), ...
    clustering(order), ...
    'VariableNames',{'Name','Sex','Party','District','Cluster'})

%%
% _Clusters with 4 ore more MPs_
for cl = unique(clustering)'
    if (sum(clustering(order) == cl) > 3)
        t_manhattan(clustering(order) == cl, 1:5);
    end
end

%%
close all;

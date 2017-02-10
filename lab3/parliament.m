%% Data Clustering: Votes of MPs

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
warning('off', 'MATLAB:mode:EmptyInput')
clc; clear; close all;

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
% For instance, let us display some of these labels  for the 10 first MPs.
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
% grid-like fashion. The neighborhood of a unit is defined by thresholding 
% the manhattan distance between the unit and the others.
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
% <include>neighborhood2.m</include>
%

%%
figure;

% *Example 1*
% Let us consider a first example with the grid $side \times side$, where 
% the winner is, say, the $27:th$ unit and the neighbourhood is of 2.
k_winner = 27;
radius = 2;
k_neighbors = neighborhood2(k_winner, radius, side_of_topologic_grid);

subplot(1,2,1);
hold on;
plot(js, is, '.b', 'MarkerSize', 15);
plot(js(k_neighbors), is(k_neighbors), '.m', 'MarkerSize', 30);
plot(js(k_winner), is(k_winner), '.r', 'MarkerSize', 40);
title(sprintf('Winner: (%d,%d)    Radius: %d', ...
    is(k_winner), js(k_winner), radius));
axis ij;
axis image;
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);

% *Example 2*
% In this example the winning unit is 67 and we consider a larger
% neighbourhood, namely of 4.
k_winner = 67;
radius = 4;
k_neighbors = neighborhood2(k_winner, radius, side_of_topologic_grid);

subplot(1,2,2);
hold on;
plot(js, is, '.b', 'MarkerSize', 15);
plot(js(k_neighbors), is(k_neighbors), '.m', 'MarkerSize', 30);
plot(js(k_winner), is(k_winner), '.r', 'MarkerSize', 40);
title(sprintf('Winner: (%d,%d)    Radius: %d', ...
    is(k_winner), js(k_winner), radius));
axis ij;
axis image;
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);

%%
%
% <include>neighborhood2_gauss.m</include>
%

figure;

% In the following, we propose to introduce a slight modification on the 
% neighbourhood obtention. So far we have assigned the same update effect
% (unitary) to all updated units, both the winner and its neighbours. We 
% now consider the usage of a gaussian distribution centered at the winner 
% unit, which tells which units are more affected by the seen point. In
% particular, the update equation is now given by 
% $ w^{new} \gets w^{old} -\eta h (x-w)$, where $h$ denotes the 
% _neighborhood_ level and decreases  with  distance  from  the winning
% unit.

% *Example 1*
k_winner = 27;
sigma = 1;
k_neighbors = neighborhood2_gauss(k_winner, sigma, side_of_topologic_grid);

subplot(1,2,1);
bar3(reshape(k_neighbors,side_of_topologic_grid,side_of_topologic_grid));
title(sprintf('Winner: (%d,%d)    Sigma: %d', ...
    is(k_winner), js(k_winner), sigma));

% *Example 2*
k_winner = 27;
sigma = 0.5;
k_neighbors = neighborhood2_gauss(k_winner, sigma, side_of_topologic_grid);

subplot(1,2,2);
bar3(reshape(k_neighbors, side_of_topologic_grid, side_of_topologic_grid));
title(sprintf('Winner: (%d,%d)    Sigma: %d', ...
    is(k_winner), js(k_winner), sigma));

%% Training
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

%% Visualizing
% *Explanation for sex*
%
% * For every unit compute the MPs associated to it, take their sex and put
% the result in a list associated to that unit. 
% * Then assign to every unit the color of the most frequent sex in its list.
% * Plot the units in the topological space with their color.

clustering = zeros(num_of_MP, 1);
sex_freq = cell(num_of_units, 1);
party_freq = cell(num_of_units, 1);
district_freq = cell(num_of_units, 1);
for mp_idx = 1:num_of_MP
    mp = votes(mp_idx, :);
    diff = repmat(mp, num_of_units, 1) - weights;
    dist = sum(diff.^2, 2);
    [~, k_winning_unit] = min(dist);
    
    clustering(mp_idx) = k_winning_unit;
    sex_freq{k_winning_unit} = [sex_freq{k_winning_unit}, sex(mp_idx)];
    party_freq{k_winning_unit} = [party_freq{k_winning_unit}, parties(mp_idx)];
    district_freq{k_winning_unit} = [district_freq{k_winning_unit}, districts(mp_idx)];
end

sex_img = -1 * ones(num_of_units, 1);
party_img = -1 * ones(num_of_units, 1);
district_img = -1 * ones(num_of_units, 1);
for k = 1:num_of_units
    sex_img(k) = mode(sex_freq{k});
    party_img(k) = mode(party_freq{k});
    district_img(k) = mode(district_freq{k});
end

figure; 
subplot(1,3,1); hold on;
for s = unique(sex)'
    scatter(js(sex_img==s), is(sex_img==s), 600, 'filled', ...
        'MarkerFaceColor', sex_colormap(s+1,:), ...
        'MarkerFaceAlpha',5/8);
end
title('Most common sex per unit');
axis ij;
axis image;
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);

subplot(1,3,2); hold on;
for p = unique(parties)'
    scatter(js(party_img==p), is(party_img==p), 600, 'filled', ...
        'MarkerFaceColor', party_colormap(p+1,:), ...
        'MarkerFaceAlpha',5/8);
end
title('Most common party per unit');
axis ij;
axis image;
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);
%legend(party_names);

subplot(1,3,3); hold on;
for d = unique(districts)'
    scatter(js(district_img==d), is(district_img==d), 600, 'filled', ...
        'MarkerFaceColor', districts_colormap(d,:), ...
        'MarkerFaceAlpha',5/8);
end
title('Most common district per unit');
axis ij;
axis image;
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);

%% Tables of clusters
%

%%
% *Association of every MP to one unit*
[~, order] = sort(clustering);
t = table(names(order), sex_labels(((sex(order) + 1)')), ...
    party_labels(((parties(order) + 1)')), ...
    districts(order), ...
    clustering(order), ...
    'VariableNames',{'Name','Sex','Party','District','Cluster'})

%%
% *Clusters with 4 ore more MPs*
for cl = unique(clustering)'
    if (sum(clustering(order) == cl) > 3)
        t(clustering(order) == cl, 1:5)
    end
end

%%
close all;

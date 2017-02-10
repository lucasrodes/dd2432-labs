%% Data Clustering: Votes of MPs

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%%
%
% This is the meaning of each variable loaded from |politics|:
%
% * |parties| = party membership of each MP
% * |sex| = sex of each MP
% * |districts| = district of each MP
% * |votes| = list of 31 votes for each MP
% * |names| = name of each MP
%
% Plus, there are some additional variables:
%
% * |sex_colormap| = colormap for sex
% * |sex_labels| = labels for sex
% * |party_colormap| = colormap for party
% * |party_labels| = labels for party

politics;
table(names(1:10), sex_labels(((sex(1:10) + 1)')), ...
    party_labels(((parties(1:10) + 1)')), districts(1:10), ...
    'VariableNames', {'Name', 'Sex', 'Party', 'District'})
%%
% *Setup*
% We will work with a 2D topology in which the units are connected in a
% grid-like fashion.

% System parameters
eta = 0.2;
num_of_epochs = 50;
side_of_topologic_grid = 10;
num_of_units = side_of_topologic_grid^2;
[num_of_MP, num_of_votes] = size(votes);
weights = rand(num_of_units, num_of_votes);

for epoch = 1:num_of_epochs

    for mp_idx = 1:num_of_MP
        mp = votes_(mp_idx, :);
        diff = repmat(mp, num_of_units, 1) - weights;
        dist = sum(diff.^2, 2);
        [~, winning_unit] = min(dist);
        
        % Initialize update function with zeros
        update_function = zeros(num_of_units, num_of_votes);
        % row axis
        neighboor_row_min = max(1, winning_unit - 1);
        neighboor_row_max = min(side_of_topologic_grid, winning_unit + 1);
        % col axis
        neighboor_col_min = max(1, winning_unit - 1);
        neighboor_col_max = min(side_of_topologic_grid, winning_unit + 1);
        % Update update function (1s corresponding to active units)
        update_function(neighboor_row_min : neighboor_row_max, ...
            neighboor_col_min:neighboor_col_max) = 1;
        
        % Update weights
        weights = weights + update_function .* (eta * diff);
    end
end


% Define meshgrid for the output grid
[x, y] = meshgrid([1:10], [1:10]);
xpos = reshape(x, 1, 100);
ypos = reshape(y, 1, 100);


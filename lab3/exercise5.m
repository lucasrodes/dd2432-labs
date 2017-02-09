% Party information of each MP
mpparty;
% SEx of each MP
mpsex;
% District of each MP
mpdistrict;
% Votes of MP
votes;

% System parameters
num_of_epochs = 50;
num_of_units = 100;
num_of_units_per_dim = sqrt(num_of_units);

% Obtain number of features, number of input samples
[num_of_MP, num_of_votes] = size(votes_);

% Initialize weight matrix
weights = rand(num_of_units, num_of_votes);

eta = 0.2;

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
        neighboor_row_max = min(num_of_units_per_dim, winning_unit + 1);
        % col axis
        neighboor_col_min = max(1, winning_unit - 1);
        neighboor_col_max = min(num_of_units_per_dim, winning_unit + 1);
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


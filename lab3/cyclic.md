clear;
cities;

% Training parameters
epochs = 20;
units = 10;
eta = 0.2;

% Number of units
nCities = size(city,1);
nFeatures = size(city,2);

% Weight matrix
W = rand(units, nCities);

for epoch = 1:epochs
    % Decrease number of neighbours
    neighbours = epochs+1-epoch;
    
    for c = 1 : nCities
        % Find winner unit
        city = cities(c,:);
        D = repmat(city, units, 1) - W;
        [d_winner, k_winner] = min(sum(D.^2));
        
        % Neighbours of winner unit
        k = ones(units, 1);
        k(1:k_winner-neighbours) = 0;
        k(k_winner+neighbours:end) = 0;
        
        % Update the weights only of winners + neighbours
        W = W + repmat(k, 1, nAttributes).*(eta*D);
    end
    
end

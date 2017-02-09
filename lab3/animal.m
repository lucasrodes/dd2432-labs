%% Topological Ordering of Animal Species

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% Introduction
% *Units*
% If we were to use RBF units, we would have for every unit an activation:
%
% $$RBF(in, w) = \exp\left( \frac{||in-w||^2}{\sigma} \right)$$
%
% and the winner unit would be the one that has the highest activation
% value.
%
% Here we use simpler units, based on distance only:
%
% $$f(in, w) = ||in-w||^2$$
%
% and the winner unit is the one that has the lowest activation value.
% 
% *Neighborood*
% We are using a one dimensional topology, i.e. the neurons are arranged
% sequentially one after the other and given a winner, we will update all
% the units that are closer than a certain number of 'hops' to it.
%
% $$w \leftarrow w + \eta (in-w)$$
%

% TODO add stuff about "the neuron that wins keeps winning"

%% Algorithm
% *Setup*
% The weights matrix contains a 84D vector for every one of the 100 units

animals;
num_of_epochs = 20;
eta = 0.2;
[num_of_animals, num_of_features] = size(props); 
num_of_units = 100;
weights = rand(num_of_units, num_of_features);

%%
% *Training*

for epoch = 1:num_of_epochs
    neighborood_size = (num_of_epochs - epoch + 1)*2;
    for animal_idx = 1:num_of_animals
        p = props(animal_idx, :);
        diff = repmat(p, num_of_units, 1) - weights;
        dist = sum(diff.^2, 2);
        [~, winning_unit] = min(dist);
        
        % Update function
        update_function = zeros(num_of_units, num_of_features);
        neighboor_min = max(1, winning_unit - neighborood_size);
        neighboor_max = min(num_of_units, winning_unit + neighborood_size);
        update_function(neighboor_min : neighboor_max, :) = 1;
        
        % Update weights
        weights = weights + update_function .* (eta * diff);
    end
end

%%
% *Result*
clustering = zeros(32, 1);
for animal_idx = 1:num_of_animals
        p = props(animal_idx, :);
        diff = repmat(p, num_of_units, 1) - weights;
        dist = sum(diff.^2, 2);
        [~, winning_unit] = min(dist);
        clustering(animal_idx) = winning_unit;
end

[~, order] = sort(clustering);
table(snames(order)', clustering(order),'VariableNames',{'Animal', 'Cluster'})

figure; 
imagesc(props(:,:)); 
title('Feature map of the animals (unsorted)');
grid on; 
grid minor; 
set(gca,'ytick',1:32);
set(gca,'yticklabels',snames);
set(gca,'xtick',1:4:84);

figure;
imagesc(props(order,:)); 
title('Feature map of the animals (sorted)');
grid on; 
grid minor; 
set(gca,'ytick',1:32);
set(gca,'yticklabels',snames(order));
set(gca,'xtick',1:4:84);
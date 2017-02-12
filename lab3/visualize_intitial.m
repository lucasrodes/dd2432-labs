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
    % Obtain the closest unit to mp
    mp = votes(mp_idx, :);
    diff = repmat(mp, num_of_units, 1) - weights;
    dist = sum(diff.^2, 2);
    [~, k_winning_unit] = min(dist);
    
    % Assign mp attributes to the found unit
    clustering(mp_idx) = k_winning_unit;
    sex_freq{k_winning_unit} = [sex_freq{k_winning_unit}, sex(mp_idx)];
    party_freq{k_winning_unit} = [party_freq{k_winning_unit}, parties(mp_idx)];
    district_freq{k_winning_unit} = [district_freq{k_winning_unit}, districts(mp_idx)];
end

sex_img = -1 * ones(num_of_units, 1);
party_img = -1 * ones(num_of_units, 1);
district_img = -1 * ones(num_of_units, 1);
% Obtain most common attribute within each unit cluster
for k = 1:num_of_units
    sex_img(k) = mode(sex_freq{k});
    party_img(k) = mode(party_freq{k});
    district_img(k) = mode(district_freq{k});
end

figure; 

% Plot sex attribute
subplot(1,3,1); hold on;
for s = unique(sex)'
    scatter(js(sex_img==s), is(sex_img==s), 600, 'filled', ...
        'MarkerFaceColor', sex_colormap(s+1,:), ...
        'MarkerFaceAlpha',5/8);
end
title('Most common sex per unit', 'Interpreter', 'latex', 'FontSize', 16);
axis ij;
axis image;
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);

% Plot party attribute
subplot(1,3,2); hold on;
for p = unique(parties)'
    scatter(js(party_img==p), is(party_img==p), 600, 'filled', ...
        'MarkerFaceColor', party_colormap(p+1,:), ...
        'MarkerFaceAlpha',5/8);
end
title('Most common party per unit', 'Interpreter', 'latex', 'FontSize', 16);
axis ij;
axis image;
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);
%legend(party_names);

% Plot district attribute
subplot(1,3,3); hold on;
for d = unique(districts)'
    scatter(js(district_img==d), is(district_img==d), 600, 'filled', ...
        'MarkerFaceColor', districts_colormap(d,:), ...
        'MarkerFaceAlpha',5/8);
end
title('Most common district per unit', 'Interpreter', 'latex', 'FontSize', 16);
axis ij;
axis image;
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
xlim([0, side_of_topologic_grid+1]);
ylim([0, side_of_topologic_grid+1]);

suptitle('SOM for different attributes, using Manhattan Distance neighbourhood');
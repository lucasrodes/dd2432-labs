%% Sparse position

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% First test - study the maximum saving capacity of the network using the bias

clc;clear;close all;

N = 100;
P = 100;
biais = 0.0;
% Create P patterns with a biais 
all_patterns = round(rand(P, N));

%We need to compute the mean of all_patterns
[pat, N] = size(all_patterns);
m = sum(sum(all_patterns))/(N*pat);

rng(1);
percentage_vec =[];
% patterns = all_patterns(1:P, :) - m;
% w = (patterns)'*(patterns);
for P = 1:size(all_patterns, 1)
    patterns = all_patterns(1:P,:);
    w_Bias = train_weights(patterns - m,false,true);
    % Transform w gto resist to noise
    %w = w-diag(diag(w));
    saved = 0;
    for original_pat = patterns'
        reconstructed_pat_Bias = evolve_net(w_Bias, original_pat,false,true);
        if sum(abs(original_pat'-reconstructed_pat_Bias)) == 0
            saved = saved + 1;
        end
    end
    % Add percentage of good pattern stored
    percentage_vec = [percentage_vec saved*100/P];
end

plot(0:pat-1, percentage_vec, 'b+-');

%%
close all;
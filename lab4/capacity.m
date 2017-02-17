%% Capacity

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% Our procedure
%
% * Take a network of N=1024 neurons
% * Train the network on P patterns
% * For every training pattern p:
%   * Consider 10 distorted versions of it, each one with noise_quantity pixels flipped
%   * Check how many of the distorted patterns get correctly reconstructed
% * Express the performance of the network as #correctly reconstructed/
% #patterns


%% Test 1
pict;
all_patterns = [p1; p2; p3; p4; p5; p6; p7; p8; p9];
N = 1024;
noisy_pixels = 100;
repetitions = 50;

rng(1);
performances = [];
for P = 1:size(all_patterns, 1)
    patterns = all_patterns(1:P, :);
    w = train_weights(patterns);
    
    successes = 0;
    for original_pat = patterns'
        for i=1:repetitions
            distorted_pat = flip_img(original_pat', noisy_pixels)';
            reconstructed_pat = evolve_net(w, distorted_pat);
            if sum(original_pat~=reconstructed_pat)<10
                successes = successes + 1;
            end
        end
    end
    
    performance = successes / (P * repetitions);
    performances = [performances, performance];
end

figure;
plot(1:size(all_patterns, 1), performances);

%% Test 2
rng(1);
all_patterns = ones(25, 1024);
figure;
for i = 1:25
    pat = all_patterns(i, :);
    pat = reshape(pat, 32, 32);
    xi = randi(32);
    xf = randi(32);
    yi = randi(32);
    yf = randi(32);
    xi = min(xi, xf);
    xf = max(xi, xf);
    yi = min(yi, yf);
    yf = max(yi, yf);
    pat(xi:xf, yi:yf) = -1;
    pat = reshape(pat, 1, 1024);
    all_patterns(i, :) = pat;
    subplot(5,5,i);
    vis(pat); 
    axis off;
end

noisy_pixels = 20;
repetitions = 10;

rng(1);
performances = [];
for P = 2:size(all_patterns, 1)
    patterns = all_patterns(1:P, :);
    w = train_weights(patterns);
    
    successes = 0;
    for original_pat = patterns'
        for i=1:repetitions
            distorted_pat = flip_img(original_pat', noisy_pixels)';
            reconstructed_pat = evolve_net(w, distorted_pat);
            if sum(original_pat~=reconstructed_pat)<5
                successes = successes + 1;
            end
        end
    end
    
    performance = successes / (P * repetitions);
    performances = [performances, performance];
end

figure;
plot(2:size(all_patterns, 1), performances);
ylim([0, 1]);

%% Test 3
rng(1);
all_patterns = sgn(randn(290, 1024));
N = 1024;
noisy_pixels = 20;
repetitions = 5;

performances = [];
for P = 70:15:size(all_patterns, 1)
    patterns = all_patterns(1:P, :);
    w = train_weights(patterns);
    
    successes = 0;
    for original_pat = patterns'
        for i=1:repetitions
            distorted_pat = flip_img(original_pat', noisy_pixels)';
            reconstructed_pat = evolve_net(w, distorted_pat);
            if sum(original_pat~=reconstructed_pat)<5
                successes = successes + 1;
            end
        end
    end
    
    performance = successes / (P * repetitions);
    performances = [performances, performance];
end

figure;
plot(70:15:size(all_patterns, 1), performances);
grid on;
title('Network performance');
xlabel('Number of training patterns');
ylabel('Percent of reconstructed patterns');

%% Test 4
rng(1);
all_patterns = sgn(randn(300, 100));
N = 100;
noisy_pixels = 20;
repetitions = 2;

performances = [];
stables = [];
for P = 1:size(all_patterns, 1)
    patterns = all_patterns(1:P, :);
    w = train_weights(patterns);
    
    successes = 0;
    stable_patterns = 0;
    for original_pat = patterns'
        % try if one original pattern remains stable after one iter
        after_one_iter = sgn(w * original_pat);
        if isequal(after_one_iter, original_pat)
            stable_patterns = stable_patterns + 1;
        end
        
        % try if a distorted pattern remains stable after one iter
        for i=1:repetitions
            distorted_pat = flip_img(original_pat', noisy_pixels)';
            reconstructed_pat = evolve_net(w, distorted_pat);
            if sum(original_pat~=reconstructed_pat)<5
                successes = successes + 1;
            end
        end
    end
    
    performance = successes / (P * repetitions);
    performances = [performances, performance];
    stables = [stables, stable_patterns/P];
end

figure;

subplot(1,2,1);
plot(1:size(all_patterns, 1), performances);
grid on;
title('Network performance');
xlabel('Number of training patterns');
ylabel('Percent of reconstructed patterns');

subplot(1,2,2);
plot(1:size(all_patterns, 1), stables);
grid on;
title('Network performance');
xlabel('Number of training patterns');
ylabel('Percent of training patterns that are stable');
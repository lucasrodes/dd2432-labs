%% Capacity

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% Our procedure
%
% <html>
% <ol>
% <li>Take a network of N neurons</li>
% <li>Train the network on P patterns</li>
% <li>For every training pattern p:
% <ol>
%   <li>Consider several distorted versions of it, each one with 
%   noise_quantity pixels flipped</li>
%   <li>Check how many of the distorted patterns get correctly reconstructed
%   (allow for a few wrong pixels)</li>
% </ol></li>
% <li>Express the denoising performance of the network as #correctly
% reconstructed / #attempted patterns</li>
% <li>Express the stability of the network as #stable patterns / #patterns</li>
% </ol>
% </html>
%

%% 9 image patterns
% Not working very well after 4 patterns, due to the imbalance between +1
% and -1 pixels in the images.
clear;  pict;
all_patterns = [p1; p2; p3; p4; p5; p6; p7; p8; p9];
print_pattern_stat(all_patterns);
noisy_pixels = 100;
repetitions = 20;

figure;
for i=1:9
    subplot(3,3,i);
    vis(all_patterns(i, :));
    title(sprintf('Pattern %d',i), 'Interpreter', 'latex', 'fontsize',16);
end

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
            if sum(original_pat~=reconstructed_pat)<5
                successes = successes + 1;
            end
        end
        figure;
        vis(reconstructed_pat)
        title(sprintf('Trained with %d patterns, reconstructing noisy pattern %d',P, original_pat), ...
            'Interpreter', 'latex', 'fontsize',16);
    end
    
    performance = successes / (P * repetitions);
    performances = [performances, performance];
end

figure;
plot(1:size(all_patterns, 1), performances);
ylim([0, 1]);
grid on;
title('Denoising of the memories', 'Interpreter', 'latex', 'fontsize',16);
xlabel('Number of training patterns', 'Interpreter', 'latex', 'fontsize',16);
ylabel('Percent of denoised patterns', 'Interpreter', 'latex', 'fontsize',16); 

suptitle('9 image patterns');
%% 25 randomly generated rectangular patterns
% Same issues as before, due to the imbalance between +1
% and -1 pixels in the images.
clear; rng(1);
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
    title(sprintf('Pattern %d',i), 'Interpreter', 'latex', 'fontsize',16);
    axis off;
end

print_pattern_stat(all_patterns);
noisy_pixels = 100;
repetitions = 10;

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
grid on;
title('Denoising of the memories', 'Interpreter', 'latex', 'fontsize',16);
xlabel('Number of training patterns', 'Interpreter', 'latex', 'fontsize',16);
ylabel('Percent of denoised patterns', 'Interpreter', 'latex', 'fontsize',16);

suptitle('25 randomly generated rectangular patterns');
%% 290 zero centered random patterns
% Now instead of images, we train the network on many patterns that are
% randomly generated from a zero centered distribution. This means that the
% probability of a pixel being 1 or -1 in any of the training patterns is
% 0.5.
%
% With zero centered patterns the performances of the network are actually
% similar to the empirical 0.138N limit.
clear; rng(1);
all_patterns = sgn(randn(290, 1024));
print_pattern_stat(all_patterns);
noisy_pixels = 100;
repetitions = 3;

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
title('Denoising of the memories', 'Interpreter', 'latex', 'fontsize',16);
xlabel('Number of training patterns', 'Interpreter', 'latex', 'fontsize',16);
ylabel('Percent of denoised patterns', 'Interpreter', 'latex', 'fontsize',16);

suptitle('290 zero centered random patterns');
%% 300 zero centered random patterns
% Same thing happens with a smaller network (note that the number of noisy 
% pixels and the similarity threshold have been lowered accordingly).
clear; rng(1);
all_patterns = sgn(randn(300, 100));
print_pattern_stat(all_patterns);
N = 100;
noisy_pixels = 30;
repetitions = 3;

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
            if sum(original_pat~=reconstructed_pat)<3
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
title('Denoising of the memories', 'Interpreter', 'latex', 'fontsize',16);
xlabel('Number of training patterns', 'Interpreter', 'latex', 'fontsize',16);
ylabel('Percent of denoised patterns', 'Interpreter', 'latex', 'fontsize',16); 

subplot(1,2,2);
plot(1:size(all_patterns, 1), stables);
grid on;
title('Stability of the memories', 'Interpreter', 'latex', 'fontsize',16);
xlabel('Number of training patterns', 'Interpreter', 'latex', 'fontsize',16);
ylabel('Percent of training patterns that are stable', 'Interpreter', 'latex', 'fontsize',16);

suptitle('300 zero centered random patterns');
%% 300 zero centered random patterns + diagonal suppression
% With the diagonal suppressed the network is much less prone to remain
% stable in the configuration it is. 
%
% This can be seen if we compare the current stability plot with the one above: 
%
% *now after the breaking point no memorized pattern is stable
% * previously many patterns remained stable after the breaking point, 
% but just because the network was less likely to change configuration, 
% not because the pattern was a meaningful energy minimum.
%
clear; rng(1);
all_patterns = sgn(randn(300, 100));
print_pattern_stat(all_patterns);
noisy_pixels = 30;
repetitions = 3;

performances = [];
stables = [];
for P = 1:size(all_patterns, 1)
    patterns = all_patterns(1:P, :);
    w = train_weights(patterns, true);
    
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
            if sum(original_pat~=reconstructed_pat)<3
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
title('Denoising of the memories', 'Interpreter', 'latex', 'fontsize',16);
xlabel('Number of training patterns', 'Interpreter', 'latex', 'fontsize',16);
ylabel('Percent of denoised patterns', 'Interpreter', 'latex', 'fontsize',16);

subplot(1,2,2);
plot(1:size(all_patterns, 1), stables);
grid on;
title('Stability of the memories', 'Interpreter', 'latex', 'fontsize',16);
xlabel('Number of training patterns', 'Interpreter', 'latex', 'fontsize',16);
ylabel('Percent of training patterns that are stable', 'Interpreter', 'latex', 'fontsize',16);

suptitle('300 zero centered random patterns + diagonal suppression');
%% 300 NON zero centered random patterns + diagonal suppression
% It is worse than the previous in denoising and stability, because the
% patterns are not zero centered.
clear; rng(1);
all_patterns = sgn(0.5 + randn(300, 100));
print_pattern_stat(all_patterns);
noisy_pixels = 30;
repetitions = 3;

performances = [];
stables = [];
for P = 1:size(all_patterns, 1)
    patterns = all_patterns(1:P, :);
    w = train_weights(patterns, true);
    
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
            if sum(original_pat~=reconstructed_pat)<3
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
title('Denoising of the memories', 'Interpreter', 'latex', 'fontsize',16);
xlabel('Number of training patterns', 'Interpreter', 'latex', 'fontsize',16);
ylabel('Percent of denoised patterns', 'Interpreter', 'latex', 'fontsize',16);

subplot(1,2,2);
plot(1:size(all_patterns, 1), stables);
grid on;
title('Stability of the memories', 'Interpreter', 'latex', 'fontsize',16);
xlabel('Number of training patterns', 'Interpreter', 'latex', 'fontsize',16);
ylabel('Percent of training patterns that are stable', 'Interpreter', 'latex', 'fontsize',16);

suptitle('300 NON zero centered random patterns + diagonal suppression');
%%
close all;
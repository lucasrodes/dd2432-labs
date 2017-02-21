%% Sparse position

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;

%% First test - study the maximum saving capacity of the network using the bias

clc;clear;

N = 200;
P = 100;
biais = 0.0;
% Create P patterns with a biais 
all_patterns = round(rand(P, N));

%We need to compute the mean of all_patterns
[pat, N] = size(all_patterns);
m = sum(sum(all_patterns))/(N*pat);

rng(1);
percentage_vec =[];

for P = 1:size(all_patterns, 1)
    patterns = all_patterns(1:P,:);
    w_Bias = train_weights(patterns - m,false,true);

    saved = 0;
    for original_pat = patterns'
        reconstructed_pat_Bias = evolve_net(w_Bias, original_pat,[],false,true,false);
        if sum(abs(original_pat'-reconstructed_pat_Bias)) == 0
            saved = saved + 1;
        end
    end
    % Add percentage of good pattern stored
    percentage_vec = [percentage_vec saved*100/P];
end

plot(0:pat-1, percentage_vec, 'b+-');

%% First experiment: 
% 10% activity

clc;clear;

N = 200;
P = 100;

act_perc = 0.1;

%lets play with the value of the bias
bias = 0;

% Create P patterns with only a 10% of activation
all_patterns = zeros(P, N);

%Generate a image with a certain percentage of activation
for i=1:P
    index = randperm(N);
    index = index(1:round(N*act_perc));
    all_patterns(i,index) = 1;
end

%We need to compute the mean of all_patterns
[pat, N] = size(all_patterns);
m = sum(sum(all_patterns))/(N*pat);

rng(1);

%
Max_bias_val = 15;
step_val = 3;

count = 0;

figure;
for bias=0:step_val:Max_bias_val
    percentage_vec =[];
    for P = 1:size(all_patterns, 1)
        patterns = all_patterns(1:P,:);
        w_Bias = train_weights(patterns - m,false,true);
        % Transform w gto resist to noise
        %w = w-diag(diag(w));
        saved = 0;
        for original_pat = patterns'
            reconstructed_pat_Bias = evolve_net(w_Bias, original_pat,[],false,true,false,bias);
            if sum(abs(original_pat'-reconstructed_pat_Bias)) == 0
                saved = saved + 1;
            end
        end
        % Add percentage of good pattern stored
        percentage_vec = [percentage_vec saved*100/P];
    end
    hold all
    plot(0:pat-1, percentage_vec,'LineWidth',2)
    count = count + 1;
    legendInfo{count} = ['bias = ' num2str(bias)];
end
legend( legendInfo)
hold off

%% Second experiment
%The objective now is to study the performance when the activity percentage
%is even smaller

clc;clear;

N = 200;
P = 100;

rng(1);

%Bias
Max_bias_val = 15;
step_val = 3;

%Act percentage 
initial_val = 0.01;
step_act = 0.01;
final_act_val = 0.05


for bias = 0:step_val:Max_bias_val
    
    figure;
    count = 0;
    all_patterns = zeros(P, N);
    for act_perc = initial_val:step_act:final_act_val
        %Generate a image with a certain percentage of activation
        for i=1:P
            index = randperm(N);
            index = index(1:round(N*act_perc));
            all_patterns(i,index) = 1;
        end

        %We need to compute the mean of all_patterns
        [pat, N] = size(all_patterns);
        m = sum(sum(all_patterns))/(N*pat);


        percentage_vec =[];
        for P = 1:size(all_patterns, 1)
            patterns = all_patterns(1:P,:);
            w_Bias = train_weights(patterns - m,false,true);
            % Transform w gto resist to noise
            %w = w-diag(diag(w));
            saved = 0;
            for original_pat = patterns'
                reconstructed_pat_Bias = evolve_net(w_Bias, original_pat,[],false,true,false,bias);
                if sum(abs(original_pat'-reconstructed_pat_Bias)) == 0
                    saved = saved + 1;
                end
            end
            % Add percentage of good pattern stored
            percentage_vec = [percentage_vec saved*100/P];
        end
        hold all
        plot(0:pat-1, percentage_vec,'LineWidth',2)
        count = count + 1;
        legendInfo{count} = ['Activation Percentage = ' num2str(act_perc)];
    end
    legend( legendInfo)
    legendInfo = [];
    title(sprintf('Learning capacity for bias = %d',bias));
    xlabel('Number of patterns');
    ylabel(' Percentage learnt ');
    hold off
end

%% Third experiment - 3D correlation

clc;clear;

N = 200;
P = 100;

rng(1);

size_m = 16;
%Bias
Max_bias_val = 16;
step_val = (Max_bias_val)/size_m;

%Act percentage 
initial_val = 0.01;
final_act_val = 0.17
step_act = -(initial_val-final_act_val)/size_m;


bias_vec = [];
per_vec = [];
learning_capacity_vec = zeros(size_m,size_m);

%Counters for plot building
count_bias = 1;
count_per = 1; 

for bias = 0:step_val:Max_bias_val
    bias_vec(count_bias) = bias;
    all_patterns = zeros(P, N);
    for act_perc = initial_val:step_act:final_act_val
        per_vec(count_per) = act_perc;
        %Generate a image with a certain percentage of activation
        for i=1:P
            index = randperm(N);
            index = index(1:round(N*act_perc));
            all_patterns(i,index) = 1;
        end

        %We need to compute the mean of all_patterns
        [pat, N] = size(all_patterns);
        m = sum(sum(all_patterns))/(N*pat);


        percentage_vec =[];
        for P = 1:size(all_patterns, 1)
            patterns = all_patterns(1:P,:);
            w_Bias = train_weights(patterns - m,false,true);
            % Transform w gto resist to noise
            %w = w-diag(diag(w));
            saved = 0;
            for original_pat = patterns'
                reconstructed_pat_Bias = evolve_net(w_Bias, original_pat,[],false,true,false,bias);
                if sum(abs(original_pat'-reconstructed_pat_Bias)) == 0
                    saved = saved + 1;
                end
            end
            % Add percentage of good pattern stored
            percentage_vec = [percentage_vec saved*100/P];
            if (saved*100/P) == 100
                saved_ant = P;
            else
                learning_capacity_vec(count_bias, count_per) = saved_ant;
            end
        end
        saved_ant = P;
        count_per = count_per + 1;
    end
    count_per = 1;
    count_bias = count_bias + 1;
end

surf(bias_vec,per_vec,learning_capacity_vec);
title('Relation between Bias, Activity level and Learning capacity')
xlabel('Bias');
ylabel(' Activity level (%) ');
zlabel(' Learning capacity ');



%%
close all;
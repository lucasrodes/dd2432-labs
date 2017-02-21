%% Sparse position

%%
set(0, 'DefaultFigurePosition', get(0,'screensize'));
clc; clear; close all;
addpath('provided_code');
%% First experiment: 
%The objective is to study how the addition of a bias term along with the
%use of binary representation instead of the [-1,1] affects the storing
%capacity of the network, when the active values of the patterns are really
%sparse.

% We are first experimenting how does the capacity evolves depending on the
% value of eta for a sparse value of 10% activity 

%Clean workspace
clc;clear;

%Number of cells in every pattern
N = 200;
%Number of patterns.
P = 100;

%Active proportion o cells
act_perc = 0.1;

% Create P patterns with only a 10% of activation
all_patterns = zeros(P, N);

% Generate patterns with a 10% of activation
for i=1:P
    index = randperm(N);
    index = index(1:round(N*act_perc));
    all_patterns(i,index) = 1;
end

% We need to compute the mean of all_patterns
[pat, N] = size(all_patterns);
% Average value of the cells in all patterns
m = sum(sum(all_patterns))/(N*pat);

% Randome seed, so all the experiments are done within the same random
% value
rng(1);

%Max bias value that will be use
Max_bias_val = 15;

%Step between bias values during the experimentation
step_val = 3;

%Counter used to plot a nice legeng
count = 0;

figure;

%We iterate the code for different values of bias starting with 0
for bias=0:step_val:Max_bias_val
    %Storing percentage vector
    percentage_vec =[];
    for P = 1:size(all_patterns, 1)
        patterns = all_patterns(1:P,:);
        %We train the network 
        %Inputs - train_weights( Input_patterns - mean value, supress diagonal = false , training with bias term = true);
        w_Bias = train_weights(patterns - m,false,true);
        %Number of saved patterns
        saved = 0;
        for original_pat = patterns'
            %We check if the patterns were stored checking if the system is
            %able to obtained the correct pattern when the pattern is the
            %input in only one iteration.
            %Inputs - reconstructed_pat_Bias = evolve_net(weights, patterns,value used in capacity.m = [],...
            % sequential evolution = false,bias term = true, convergenge check = false as we only want one iteration ...
            % bias value);
            reconstructed_pat_Bias = evolve_net(w_Bias, original_pat,-1,false,true,false,bias);
            % Add one the the saved patterns counter
            if sum(abs(original_pat'-reconstructed_pat_Bias)) == 0
                saved = saved + 1;
            end
        end
        % Add percentage of good pattern stored to the vector
        percentage_vec = [percentage_vec saved*100/P];
    end
    %Plotting the results for every value of bias
    hold all
    plot(0:pat-1, percentage_vec,'LineWidth',2);
    grid on;
    count = count + 1;
    legendInfo{count} = ['bias = ' num2str(bias)];
end

%Legend and labels
legend( legendInfo, 'Interpreter','latex', 'fontsize', 16);
title('Learning capacity evolution depending on the bias value', ...
    'Interpreter','latex', 'fontsize', 16);
xlabel('Number of patterns', 'Interpreter','latex', 'fontsize', 16);
ylabel(' Percentage learnt ', 'Interpreter','latex', 'fontsize', 16);
hold off

%% Second experiment
%The objective is to study how the addition of a bias term along with the
%use of binary representation instead of the [-1,1] affects the storing
%capacity of the network, when the active values of the patterns are really
%sparse.

% Now we want to check how does the storing capacity evolves depending on
% the sparse parameter and the bias term. 

%Clean workspace
clc;clear;

%Number of cells in every pattern
N = 200;
%Number of patterns.
P = 100;

% Randome seed, so all the experiments are done within the same random
% value
rng(1);

%Max bias value that will be use
Max_bias_val = 15;

%Step between bias values during the experimentation
step_val = 3;

% We will experiment with different sparse values
%Initial value
initial_val = 0.01;

%Step in the sparse value between experiments
step_act = 0.01;

%Final value
final_act_val = 0.05

%For every bias value
for bias = 0:step_val:Max_bias_val
    %We want a figure for every bias value
    figure;
    %Counter used for legend plotting
    count = 0;
    all_patterns = zeros(P, N);
    %We iterate for the different sparse values
    for act_perc = initial_val:step_act:final_act_val
        % Generate pattern with a certain percentage of activation
        for i=1:P
            index = randperm(N);
            index = index(1:round(N*act_perc));
            all_patterns(i,index) = 1;
        end

        % We need to compute the mean of all_patterns
        [pat, N] = size(all_patterns);
        % Average value of the cells in all patterns
        m = sum(sum(all_patterns))/(N*pat);

        %As before, we need to store the storing capacity with the
        %different conditions
        percentage_vec =[];
        for P = 1:size(all_patterns, 1)
            patterns = all_patterns(1:P,:);
            %We train the network 
            %Inputs - train_weights( Input_patterns - mean value, supress diagonal = false , training with bias term = true);
            w_Bias = train_weights(patterns - m,false,true);
            %Number of saved patterns
            saved = 0;
            for original_pat = patterns'
                %We check if the patterns were stored checking if the system is
                %able to obtained the correct pattern when the pattern is the
                %input in only one iteration.
                %Inputs - reconstructed_pat_Bias = evolve_net(weights, patterns,value used in capacity.m = [],...
                % sequential evolution = false,bias term = true, convergenge check = false as we only want one iteration ...
                % bias value);
                reconstructed_pat_Bias = evolve_net(w_Bias, original_pat,-1,false,true,false,bias);
                % Add one the the saved patterns counter
                if sum(abs(original_pat'-reconstructed_pat_Bias)) == 0
                    saved = saved + 1;
                end
            end
            % Add percentage of good pattern stored
            percentage_vec = [percentage_vec saved*100/P];
        end
        %Plotting
        hold all
        plot(0:pat-1, percentage_vec,'LineWidth',2)
        count = count + 1;
        legendInfo{count} = ['Activation Percentage = ' num2str(act_perc)];
    end

    %Legend
    grid on;
    legend( legendInfo,'interpreter','latex','fontsize',16);
    legendInfo = [];
    title(sprintf('Learning capacity for bias = %d',bias),...
        'interpreter','latex','fontsize',16);
    xlabel('Number of patterns');
    ylabel(' Percentage learnt ');
    hold off
end

%% Third experiment - 3D correlation
%The objective is to study how the addition of a bias term along with the
%use of binary representation instead of the [-1,1] affects the storing
%capacity of the network, when the active values of the patterns are really
%sparse.

% Now we want to study the relation between the bias, the sparse value and
% the storing capacity all together

%Clean workspace
clc;clear;

%Number of cells in every pattern
N = 200;
%Number of patterns.
P = 100;

%Number of values we are experimenting with
size_m = 16;

%Max value for the bias
Max_bias_val = 16;

%Step in bias values between experiments. It is done so that the number of
%iterations is the sime as size_m
step_val = (Max_bias_val)/size_m;

% Max value of the sparse value 
initial_val = 0.01;

%Final value 
final_act_val = 0.17;

%Step in sparse values between experiments. It is done so that the number of
%iterations is the sime as size_m
step_act = -(initial_val-final_act_val)/size_m;

%Vectors that will store the data used as inputs in the experiments (Bias
%and sparse values) so that we can plot the results
bias_vec = [];
per_vec = [];

%Matrix that will store the storing capacity for the different values of
%bias and sparse
learning_capacity_vec = zeros(size_m,size_m);

% Counters for plotting
count_bias = 1;
count_per = 1; 

%We iterate for every different bias value
for bias = 0:step_val:Max_bias_val
    %We store the value for future plotting
    bias_vec(count_bias) = bias;
    all_patterns = zeros(P, N);
    %We iterate for every sparse value
    for act_perc = initial_val:step_act:final_act_val
        %We store the value for future plotting
        per_vec(count_per) = act_perc;
        %Generate pattern with a certain percentage of activation
        for i=1:P
            index = randperm(N);
            index = index(1:round(N*act_perc));
            all_patterns(i,index) = 1;
        end

        % % We need to compute the mean of all_patterns
        [pat, N] = size(all_patterns);
        % Average value of the cells in all patterns
        m = sum(sum(all_patterns))/(N*pat);

        %Vector to store the different store capacity values
        percentage_vec =[];
        for P = 1:size(all_patterns, 1)
            
            patterns = all_patterns(1:P,:);
            %We train the network 
            %Inputs - train_weights( Input_patterns - mean value, supress diagonal = false , training with bias term = true);
            w_Bias = train_weights(patterns - m,false,true);
            %Number of saved patterns
            saved = 0;
            for original_pat = patterns'
                %We check if the patterns were stored checking if the system is
                %able to obtained the correct pattern when the pattern is the
                %input in only one iteration.
                %Inputs - reconstructed_pat_Bias = evolve_net(weights, patterns,value used in capacity.m = [],...
                % sequential evolution = false,bias term = true, convergenge check = false as we only want one iteration ...
                % bias value);
                reconstructed_pat_Bias = evolve_net(w_Bias, original_pat,[],false,true,false,bias);
                % Add one the the saved patterns counter
                if sum(abs(original_pat'-reconstructed_pat_Bias)) == 0
                    saved = saved + 1;
                end
            end
            % Add percentage of good pattern stored
            percentage_vec = [percentage_vec saved*100/P];
            
            %We will save the last number of patterns for which the network
            %can store all of them
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
%Plotting
surf(bias_vec,per_vec,learning_capacity_vec);
title('Relation between Bias, Activity level and Learning capacity',...
    'Interpreter','latex', 'fontsize',16);
xlabel('Bias','Interpreter','latex', 'fontsize',16);
ylabel(' Activity level (%) ','Interpreter','latex', 'fontsize',16);
zlabel(' Learning capacity ','Interpreter','latex', 'fontsize',16);



%%
close all;
% This script runs the delta rule
clear;

% "Global" variables
SEPARABLE_DATA = 0;
NONSEPARABLE_DATA = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set to 1 for LaTeX labeling, 0 for default labeling
LATEX = 1;
% Choose data set to be used
mode = NONSEPARABLE_DATA;


if LATEX
    int = 'latex';
else
    int = 'tex';
end

switch mode
    case SEPARABLE_DATA
        % Easily separable dataset
        [patterns, targets] = sepdata;
        tit = 'Separable Data';
    otherwise
        % Non separable dataset
        [patterns, targets] = nsepdata;
        tit = 'Non-Separable Data';
end

% Size of input/output
[insize, ~] = size(patterns);
[outsize, ndata] = size(targets);

% Input matrix with ones (bias)
X = [patterns; ones(1, ndata)];

% Learning parameter
eta = 0.001;

% Random initialization of the weights
W = randn(outsize, insize+1);

% Number of epochs
epochs = 20;

for i = 0:epochs
    % Delta Rule, update weights
    delta_W = -eta*( W*X - targets)*X';
    W = W + delta_W;
        
    % Prepare plots
    p = W(1, 1:2);
    k = -W(1, size(patterns, 1)+1) / (p*p');
    l = sqrt(p*p');
    
    % Plot the results
    plot (patterns(1, targets>0), ...
        patterns(2, targets>0), '*', ...
        patterns(1, targets<0), ...
        patterns(2, targets<0), '+');
    hold on;
    plot([p(1), p(1)]*k + [-p(2), p(2)]/l, ...
        [p(2), p(2)]*k + [p(1), -p(1)]/l, 'k-', 'LineWidth', 3);
    title(tit, 'FontSize', 20,'Interpreter',int);
    xlabel('$$x_1$$','Interpreter',int,'FontSize', 20);
    ylabel('$$x_2$$','Interpreter',int,'FontSize', 20);
    h_legend = legend('+1', '-1');
    set(h_legend,'FontSize',16,'Interpreter',int, 'Location', 'southeast');
    hold off;
    axis([-2, 2, -2, 2], 'square');
    drawnow;
    pause(0.25);
end
% This script runs the delta rule
clear;

SEPARABLE_DATA = 0;
NONSEPARABLE_DAT = 1;

% Set to 1 for LaTeX labeling, 0 for default labeling
LATEX = 1;

if LATEX
    int = 'latex';
else
    int = 'tex';
end


% Choose data set to be used
mode = SEPARABLE_DATA;

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

% Initialization
eta = 0.001;
X = [patterns; ones(1, size(patterns, 2))];
W = randn(outsize, insize+1);
epochs = 20;

for i = 0:epochs
    % Delta Rule, update weights
    deltaW = -eta*( W*X - targets)*X';
    W = W + deltaW;
        
    % Prepare plots
    p = W(1, 1:2);
    k = -W(1, size(patterns, 1)+1) / (p*p');
    l = sqrt(p*p');
    
    % Plot results
    plot (patterns(1, targets>0), ...
        patterns(2, targets>0), '*', ...
        patterns(1, targets<0), ...
        patterns(2, targets<0), '+');
    hold on;
    plot([p(1), p(1)]*k + [-p(2), p(2)]/l, ...
        [p(2), p(2)]*k + [p(1), -p(1)]/l, 'k-', 'LineWidth', 3);
    title(tit, 'FontSize', 14,'Interpreter',int);
    hold off;
    axis([-2, 2, -2, 2], 'square');
    drawnow;
    pause(0.25);
end
% This script runs the delta rule
clear;

% "Global" variables
SEPARABLE_DATA = 0;
NONSEPARABLE_DATA = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set to 1 for LaTeX labeling, 0 for default labeling
LATEX = 1;
% Choose data set to be used
mode = SEPARABLE_DATA;
% Set number of hidden layers
hidden = 4;
% Set number of epochs
epoch = 500;
% Choose learning rate
eta = 0.01;


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
alpha = 0.9;

% Initialize the weights
W = randn(hidden, insize+1);
V = randn(outsize, hidden+1);
delta_W = 0;
delta_V = 0;

error = zeros(1,epoch);
steps = 1:epoch;

for i = steps
    % 1. Forward pass
    % Hidden layer
    hin = W * X;
    hout = [phi(hin); ones(1, ndata)];

    % Output Layer
    oin = V * hout;
    out = phi(oin);

    % 2. Backward pass
    delta_o = (out - targets) .* phiprime(out);
    delta_h = (V' * delta_o) .* phiprime(hout);
    delta_h = delta_h(1:hidden, :); % remove bias term

    % 3. Weight update
    delta_W = (delta_W .* alpha) - (delta_h * X') .* (1 - alpha);
    delta_V = (delta_V .* alpha) - (delta_o * hout') .* (1 - alpha);
    W = W + delta_W .* eta;
    V = V + delta_V .* eta;
    
    % Obtain error
    error(1, i) = sum(sum(abs(sign(out) - targets) ./ 2));
    % Plot error
    plot(steps, error);
    title(['MSE per Epoch - ' tit], 'FontSize', 20,'Interpreter',int);
    xlabel('Epoch','Interpreter',int,'FontSize', 20);
    ylabel('Error','Interpreter',int,'FontSize', 20);
    pause(0.05)
end
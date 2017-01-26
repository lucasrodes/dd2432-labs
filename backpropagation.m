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
alpha = 0.9;

hidden = 4;
W = randn(hidden, insize+1);
V = randn(outsize, hidden+1);
dw = 0;
dv = 0;

eta = 0.01;
error = zeros(1,500);
steps = 1:500;
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
    delta_h = delta_h(1:hidden, :);

    % 3. Weight update
    dw = (dw .* alpha) - (delta_h * X') .* (1 - alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1 - alpha);
    W = W + dw .* eta;
    V = V + dv .* eta;
    
    error(1, i) = sum(sum(abs(sign(out) - targets) ./ 2));
    plot(steps, error);
    pause(0.05)
end
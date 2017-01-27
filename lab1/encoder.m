% The encoder problem consists on using the same input and output data and
% hence forcing the NN to obtain a compressed representation in the hidden
% layer.
clear;

hidden = 3;

LATEX = 1;
if LATEX
    int = 'latex';
else
    int = 'tex';
end

% Define input and output
patterns = eye(8) * 2 - 1;
targets = patterns;

% Size of input/output
[insize, ~] = size(patterns);
[outsize, ndata] = size(targets);

% Input matrix with ones (bias)
X = [patterns; ones(1, ndata)];
alpha = 0.9; % used to average updates and hence avoid noisy updates
    
% Initialize the weights
W = randn(hidden, insize+1);
V = randn(outsize, hidden+1);
delta_W = 0;
delta_V = 0;

eta = 0.05;
hidden = 3;

%x = eye(8) * 2 - 1;
%[x_row, x_col] = size(x);

epoch = 3000;
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
    delta_h = delta_h(1:hidden, :);

    % 3. Weight update
    delta_W = (delta_W .* alpha) - (delta_h * X') .* (1 - alpha);
    delta_V = (delta_V .* alpha) - (delta_o * hout') .* (1 - alpha);
    W = W + delta_W .* eta;
    V = V + delta_V .* eta;
    
    % Obtain error
    error(1,i) = sum(sum(abs(sign(out) - targets) ./ 2));
    % Plot error
    plot(steps, error);
    title('MSE per Epoch', 'FontSize', 20,'Interpreter',int);
    xlabel('Epoch','Interpreter',int,'FontSize', 20);
    ylabel('Error','Interpreter',int,'FontSize', 20);
    %pause(0.005)
end

% Forward pass
hin = W * X;
hout = [phi(hin); ones(1, ndata)];

oin = V * hout;
out = phi(oin);

[~, human_readable_in] = max(patterns);
human_readable_hid = hout(1:3,:) > 0;
hid_strings = [];
for i = 1:8
    hid_strings = [hid_strings; num2str(human_readable_hid(:,i)')];
end
[~, human_readable_out] = max(out);
table(human_readable_in(:), hid_strings, human_readable_out(:), 'VariableNames', {'In', 'Hidden', 'Out'})
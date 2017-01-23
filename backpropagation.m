clear;

% Linearly separable dataset
%[patterns, targets] = sepdata;

% Non linearly separable dataset
[patterns, targets] = nsepdata;

ndata = size(patterns, 2);
X = [patterns; ones(1, ndata)];
alpha = 0.9;

hidden = 4;
w = randn(hidden, size(X, 1));
v = randn(1, hidden+1);
dw = 0;
dv = 0;

eta = 0.01;
error = zeros(1,500);
steps = 1:500;
for i = steps
    % Forward pass
    hin = w * [patterns; ones(1, ndata)];
    hout = [phi(hin); ones(1, ndata)];

    oin = v * hout;
    out = phi(oin);

    % Backward pass
    delta_o = (out - targets) .* phiprime(out);
    delta_h = (v' * delta_o) .* phiprime(hout);
    delta_h = delta_h(1:hidden, :);

    % Weight update
    dw = (dw .* alpha) - (delta_h * X') .* (1 - alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1 - alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;
    
    error(1, i) = sum(sum(abs(sign(out) - targets) ./ 2));
    plot(steps, error);
    pause(0.05)
end
clear;

% Easily separable dataset
%[patterns, targets] = sepdata;

% Non separable dataset
[patterns, targets] = nsepdata;

% Initialization
eta = 0.001;
X = [patterns; ones(1, size(patterns, 2))];
w = randn(1, size(X, 1));
epochs = 20;

for i = 0:epochs
    deltaW = -eta*( w*X - targets)*X';
    w = w + deltaW;
        
    % Show
    p = w(1, 1:2);
    k = -w(1, size(patterns, 1)+1) / (p*p');
    l = sqrt(p*p');
    plot (patterns(1, find(targets>0)), ...
        patterns(2, find(targets>0)), '*', ...
        patterns(1, find(targets<0)), ...
        patterns(2, find(targets<0)), '+', ...
        [p(1), p(1)]*k + [-p(2), p(2)]/l, ...
        [p(2), p(2)]*k + [p(1), -p(1)]/l, '-');
    axis([-2, 2, -2, 2], 'square');
    drawnow;
    pause(0.5);
end
clear;

% Experiments:
% - number of samples (15 is bad, 40 is good)
n = 40;

% - number of hidden neurons (with n=40: 2 is bad, 
%                                        6 is already good, 
%                                        1000 is too much because 
%                                        many weights just go to zero)
hidden = 6;

% - number of epochs
epochs = 5000;

% Create original function
x_full=[-5:1:5]';
y_full=x_full;
z=exp(-x_full.*x_full*0.1) * exp(-y_full.*y_full*0.1)' - 0.5;
[z_row, z_col] = size(z);
ndata = z_row*z_col;

figure(1);
subplot(1,2,1);
mesh( x_full, y_full, z);
axis([-5 5 -5 5 -0.7 0.7]);

% Parse ann value
targets = reshape(z, 1, ndata);
[xx, yy] = meshgrid(x_full, y_full);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

% Randomize and select the data
permute = randperm(ndata);
% Permute data
x = patterns(:, permute);
y = targets(:, permute);
% Samples from the gaussian
x = x(:, 1:n);
y = y(:, 1:n);
[p_row, p_col] = size(x);

hold on;
scatter3(x(1,:), x(2,:), y, 'r');
hold off;

% Initialize ANN value
error = [];
alpha = 0.9;
eta = 0.15;
w = randn(hidden, p_row+1);
v = randn(1, hidden+1);
dw = 0;
dv = 0;

out = [];
for i = 1:epochs
    % Forward pass training
    hin = w * [x; ones(1, p_col)];
    hout = [phi(hin); ones(1, p_col)];

    oin = v * hout;
    out = phi(oin);

    % Backward pass
    delta_o = (out - y) .* phiprime(out);
    delta_h = (v' * delta_o) .* phiprime(hout);
    delta_h = delta_h(1:hidden, :);

    % Weight update
    dw = (dw .* alpha) - (delta_h * [x; ones(1, p_col)]') .* (1 - alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1 - alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;
    
    error = [error ; sum((y - out).^2) / sum(y .^2)];
    
    if mod(i, 10) == 0
        figure(1);
        subplot(1,2,2);
        % Forward pass test
        hin = w * [patterns; ones(1, ndata)];
        hout = [phi(hin); ones(1, ndata)];

        oin = v * hout;
        out = phi(oin);
        zz = reshape(out, z_row, z_col);
        mesh(x_full, y_full, zz);
        axis([-5 5 -5 5 -0.7 0.7]);
        title(sprintf('%d', i));
        drawnow;
    end
end

figure(2);
plot(1:epochs, error);

figure(3);
hist([v(:);w(:)],100)
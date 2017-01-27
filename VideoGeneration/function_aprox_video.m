function function_aprox_video(hidden, n, VidName, res)

% Video object
vid = VideoWriter(['Videos/',VidName]);

% Main options
vid.FrameRate = res(1);  % Default 30
vid.Quality = res(2);    % Default 75
open(vid);
    
% Set to 1 for LaTeX labeling, 0 for default labeling
LATEX = 1;
% Number of epochs
epochs = 2000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if LATEX
    int = 'latex';
else
    int = 'tex';
end

% Create original function
x_full=[-5:1:5]';
y_full=x_full;
z=exp(-x_full.*x_full*0.1) * exp(-y_full.*y_full*0.1)' - 0.5;
[z_row, z_col] = size(z);
ndata = z_row*z_col;

% Plot the function f(x,y) to approximate
figure(1);
subplot(2,2,1);
mesh( x_full, y_full, z);
xlabel('x','Interpreter',int,'FontSize', 16);
ylabel('y','Interpreter',int,'FontSize', 16);
zlabel('z','Interpreter',int,'FontSize', 16);
title(['Samples: ', num2str(n), ' -- Hidden Neurons: ', num2str(hidden)], ...
    'FontSize', 16, 'Interpreter', int);
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
[insize, p_col] = size(x);
[outsize, ~] = size(y);

% Add noise
% x = x + randn(insize, p_col);

hold on;
scatter3(x(1,:), x(2,:), y, 'r');
hold off;

% Input matrix with ones (bias)
X = [x; ones(1, p_col)];
alpha = 0.9; % used to average updates and hence avoid noisy updates

% Learning Rate
eta = 0.15;

% Initialize the weights
W = randn(hidden, insize+1);
V = randn(outsize, hidden+1);
delta_W = 0;
delta_V = 0;

error = zeros(1,epochs);
steps = 1:epochs;

for i = steps
    % 1. Forward pass training
    % Hidden layer
    hin = W * X;
    hout = [phi(hin); ones(1, p_col)];
    % Output layer
    oin = V * hout;
    out = phi(oin);

    % 2. Backward pass
    delta_o = (out - y) .* phiprime(out);
    delta_h = (V' * delta_o) .* phiprime(hout);
    delta_h = delta_h(1:hidden, :); % remove bias term

    % 3. Weight update
    delta_W = (delta_W .* alpha) - (delta_h * X') .* (1 - alpha);
    delta_V = (delta_V .* alpha) - (delta_o * hout') .* (1 - alpha);
    W = W + delta_W .* eta;
    V = V + delta_V .* eta;
    
    % Obtain the error
    error(i) = sum((y - out).^2) / sum(y .^2);
    
    if mod(i, 10) == 0
        % Forward pass test
        hin = W * [patterns; ones(1, ndata)];
        hout = [phi(hin); ones(1, ndata)];
        oin = V * hout;
        out = phi(oin);
        
        % Plot the approximation
        zz = reshape(out, z_row, z_col);
        subplot(2,2,2);
        mesh(x_full, y_full, zz);
        axis([-5 5 -5 5 -0.7 0.7]);
        title(sprintf('Iterations, %d', i),'Interpreter',int,'FontSize'...
            , 16);
        xlabel('x','Interpreter',int,'FontSize', 16);
        ylabel('y','Interpreter',int,'FontSize', 16);
        zlabel('z','Interpreter',int,'FontSize', 16);
        axis([-5 5 -5 5 -0.7 0.7]);
        drawnow;
        % Plot the error
        %figure(2);
        subplot(2,2,3);
        plot(1:epochs, error);
        grid on;
        title('MSE per epoch','Interpreter',int,'FontSize', 16);
        xlabel('Epoch','Interpreter',int,'FontSize', 16);
        ylabel('Error','Interpreter',int,'FontSize', 16);
        % Histogram of the weights
        %figure(3);
        subplot(2,2,4);
        hist([V(:);W(:)],100, 'k')
        xlabel('$$i$$','Interpreter',int,'FontSize', 16);
        ylabel('$$w_i$$','Interpreter',int,'FontSize', 16);
        title('Value of the weights','Interpreter',int,'FontSize', 16);
        F = getframe(gcf);
        writeVideo(vid,F);
    end
end

close(vid);
end



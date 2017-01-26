function Delta_video(mode)
% This script runs the delta rule
close all; 


SEPARABLE_DATA = 1;
NONSEPARABLE_DAT = 0;

%Video object
vid = VideoWriter('2D_delta.avi');

%Main options
vid.FrameRate = 10;  % Default 30
vid.Quality = 50;    % Default 75
open(vid);


% Set to 1 for LaTeX labeling, 0 for default labeling
LATEX = 1;

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
        
        %Video object
        vid = VideoWriter('2D_delta_Separable.avi');

        %Main options
        vid.FrameRate = 10;  % Default 30
        vid.Quality = 50;    % Default 75
        open(vid);
    
    otherwise
        % Non separable dataset
        [patterns, targets] = nsepdata;
        tit = 'Non-Separable Data';
        
        %Video object
        vid = VideoWriter('2D_delta_Non_ Separable.avi');

        %Main options
        vid.FrameRate = 10;  % Default 30
        vid.Quality = 50;    % Default 75
        open(vid);
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
    title(tit, 'FontSize', 16,'Interpreter',int);
    h_legend = legend('+1', '-1');
    set(h_legend,'FontSize',14,'Interpreter',int, 'Location', 'southeast');
    hold off;
    axis([-2, 2, -2, 2], 'square');
    F = getframe;
    writeVideo(vid,F);
end

close(vid);

end
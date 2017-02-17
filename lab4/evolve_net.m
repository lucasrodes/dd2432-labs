% x_start and x_end are column vectors
function [ x_end, iterations ] = evolve_net(w, x_start, sequential)
    N = size(w, 1);
    iterations = 0;
    
    if nargin < 3
        sequential = false;
    end
    if nargin < 2
        x_start = sgn(randn(N, 1));
    end
    
    if sequential
        converged = false;
        x_end = x_start;
        figure;
        while not(converged) && iterations < 10000
            converged = true;
            for i = randperm(N)
                old_x = x_end(i);
                x_end(i) = sgn(w(i,:) * x_end);
                iterations = iterations + 1;
                if mod(iterations, 10)==0
                    subplot(1,2,1);
                    vis(x_end);
                    title(sprintf('Iter: %d', iterations));
                    subplot(1,2,2);
                    hold on;
                    E = - x_end' * w * x_end;
                    plot(iterations, E, '.b');
                    xlabel('Iterations');
                    ylabel('Energy');
                    pause(0.05);
                end
                if old_x ~= x_end(i)
                    converged = false;
                end
            end
        end
        subplot(1,2,1);
        vis(x_end);
        title(sprintf('Converged iter: %d', iterations));
    else
        converged = false;
        while not(converged) && iterations < 100
            x_end = sgn(w * x_start);
            converged = isequal(x_start, x_end);
            iterations = iterations + 1;
            x_start = x_end;
        end
    end
end


% x_start and x_end are column vectors
function [ x_end, iterations ] = evolve_net(w, x_start, sequential, bias)
    N = size(w, 1);
    iterations = 0;
    
    if nargin < 4
        bias = false;
    end
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
        title(sprintf('Converged iter: %d', ions));
    else
        converged = false;
        while not(converged) && iterations < 100
            if bias == false
                x_end = sgn(w * x_start);
            else
                %Bias term is chosen to be the mean of patterns 
                biais = sum(x_start)/length(x_start);
                x_end = 0.5 + 0.5 * sgn( (x_start'*w' - biais) );
            end
            converged = isequal(x_start, x_end);
            iterations = iterations + 1;
            if bias
                x_start = x_end';
            else
                x_start = x_end;
            end
        end
    end
end


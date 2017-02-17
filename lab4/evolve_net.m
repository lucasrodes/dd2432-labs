function [ x_end, iterations ] = evolve_net( w, x_start, sequential)
    N = size(w, 1);
    iterations = 0;
    
    if nargin < 3
        sequential = false;
    end
    if nargin < 2
        x_start = sgn(randn(N, 1));
    end
    
    if sequential 
    else
        converged = false;
        while not(converged)
            x_end = sgn(w * x_start);
            converged = isequal(x_start, x_end);
            iterations = iterations + 1;
            x_start = x_end;
        end
    end
end


% x_start and x_end are column vectors
% w: is the matrix of weights (symmetric)
% sequential: Activate it if asynchronous update is wanted
% bias: Activate it to activate the bias term.
% conv_seek: if true, the code will be iterated untill convergenge. If
% false it will be only iterated once.
% bias_val: sets the bias value
function [ x_end, iterations ] = evolve_net(w, x_start, patterns, ...
    sequential, bias, conv_seek,  bias_val)


    N = size(w, 1);
    iterations = 0;
    
    if nargin < 7
        %Used when bias = true but no bias values was given
       bias_val = sum(x_start)/length(x_start);
    end
    if nargin < 6
       conv_seek = true;
    end
    if nargin < 5
        bias = false;
    end
    if nargin < 4
        sequential = false;
    end
    if nargin < 3
        patterns = 0;
    end
    if nargin < 2
        x_start = sgn(randn(N, 1));
    end
    
    if sequential
        converged = false;
        x_end = x_start;
        figure;
        E = zeros(1, 10000);
        while not(converged) && iterations < 10000
%             converged = true;
            for i = randperm(N)
%                 old_x = x_end(i);
                x_end(i) = sgn(w(i,:) * x_end);
                iterations = iterations + 1;
                E(iterations) = - x_end' * w * x_end;
                if mod(iterations, 10)==0
                    subplot(1,2,1);
                    vis(x_end);
                    title(sprintf('Iter: %d', iterations),...
                        'Interpreter', 'latex', 'Fontsize', 16);
                    subplot(1,2,2);
                    hold on;
                    
                    plot(iterations, E(iterations), '.b');
                    title('Network Energy',...
                        'Interpreter', 'latex', 'Fontsize', 16);
                    xlabel('Iterations', 'Interpreter', 'latex',...
                        'Fontsize', 16);
                    ylabel('Energy', 'Interpreter', 'latex',...
                        'Fontsize', 16);
                end
                h = sprintf('Finished, %d iterations', iterations);
                % Convergence criteria
                if iterations>1
                    s = find(diff(E(1:iterations))~=0);
                    if isempty(s)
                        s = iterations;
                    end
                    constant_energy = iterations - s(end);
                    if ~isequal(-1, patterns)
                        [ismem, id] = ismember(x_end', patterns, 'rows');
                        if ismem
                            converged = true;
                            h = sprintf('Match found with pattern p%d at iteration %d', id, iterations);
                            break;
                        elseif constant_energy>N/4
                            h = sprintf('Converged, but no match found! at iteration %d', iterations);;
                            converged = true;
                            break;
                        else
                            converged = false;
                        end
                    else
                        if constant_energy>N/4
                            h = sprintf('Converged after %d iterations', iterations);;
                            converged = true;
                            break;
                        else
                            converged = false;
                        end
                    end
                end
%                 if old_x ~= x_end(i)
%                     converged = false;
%                 end
            end
            if conv_seek == false
                return;
            end
        end
        subplot(1,2,1);
        vis(x_end);
        title(h, 'Interpreter', 'latex', 'Fontsize', 16);
    else
        converged = false;
        while not(converged) && iterations < 100 
            if bias == false
                x_end = sgn(w * x_start);
            else
                %Evolution formula for bias term
                x_end = 0.5 + 0.5 * sgn( (x_start'*w' - bias_val) );
            end
            converged = isequal(x_start, x_end);
            iterations = iterations + 1;
            if bias
                %Need as the input pattern has to be transposed
                x_start = x_end';
            else
                x_start = x_end;
            end
            if conv_seek == false
                %If we dont want to seek the convergence but only iterate
                %once, we return. 
                return;
            end
        end
    end


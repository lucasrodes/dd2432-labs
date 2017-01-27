% function [ patterns, targets ] = sepdata(  )
%
% Generates linear separable data
function [ patterns, targets ] = sepdata(  )
    A_mean = [1.0, 0.5];
    A_std = 0.5;
    B_mean = [-1.0,0.5];
    B_std = 0.5;
    classA(1,:) = randn(1,100) .* A_std + A_mean(1);
    classA(2,:) = randn(1,100) .* A_std + A_mean(2);
    classB(1,:) = randn(1,100) .* B_std + B_mean(1);
    classB(2,:) = randn(1,100) .* B_std + B_mean(2);
    
    patterns = [classA, classB];
    targets = [ones(1,100), - ones(1,100)];
    permute = randperm(200);
    patterns = patterns(:, permute);
    targets = targets(:, permute);
end


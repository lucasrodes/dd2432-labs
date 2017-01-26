function [ patterns, targets ] = nsepdata(  )
    classA(1,:) = [randn(1,50) .* 0.2 - 1.0, randn(1,50) .* 0.2 + 1.0];
    classA(2,:) = randn(1,100) .* 0.2 + 0.3;
    classB(1,:) = randn(1,100) .* 0.3 + 0.0;
    classB(2,:) = randn(1,100) .* 0.3 - 0.1;
    
    patterns = [classA, classB];
    targets = [ones(1,100), - ones(1,100)];
    permute = randperm(200);
    patterns = patterns(:, permute);
    targets = targets(:, permute);
end


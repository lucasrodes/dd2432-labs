function [ classA, classB ] = sepdata(  )
    A_mean = [1.0, 0.5];
    A_std = 0.5;
    B_mean = [-1.0,0.5];
    B_std = 0.5;
    classA(1,:) = randn(1,100) .* A_std + A_mean(1);
    classA(2,:) = randn(1,100) .* A_std + A_mean(2);
    classB(1,:) = randn(1,100) .* B_std + B_mean(1);
    classB(2,:) = randn(1,100) .* B_std + B_mean(2);


end


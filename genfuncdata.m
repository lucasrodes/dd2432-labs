function [ patterns, targets ] = genfuncdata()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    
x = -5:1:5;
y = x;
z = exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
ndata = length(x)*length(y); 

% mesh (x, y, z);

targets = reshape (z, 1, ndata);
[xx, yy] = meshgrid(x,y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];


end


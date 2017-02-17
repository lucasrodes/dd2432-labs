function y=flip_img(x,n)
y=x;
[~,index]=sort(rand(1,size(x,2)));
y(index(1:n))=-y(index(1:n));
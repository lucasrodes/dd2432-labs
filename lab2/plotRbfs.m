
rbfs = zeros(1,size(x,2));
for u = 1:units
    rbfs = rbfs + normpdf(x,m(u),var(u));
end

plot(rbfs,'k.');
function result = phiprime( x )
     temp = phi(x);
     result = ( (1 + temp) .* (1 - temp) ) / 2;
end


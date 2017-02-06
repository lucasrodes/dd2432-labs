% Function approximation with gaussian RBF

%Do iterative improvement according to the delta rule

Phi=calcPhi(x,m,sigma2);
f=feval(fun,x);
y=x;
alg='Stochastic';
iterstart=iter;
iterstop=iter+itermax;
psum=zeros(1,itermax/itersub);
while iter<iterstop
  substop=iter+itersub;
  esum=0;
  while iter<substop
    iter=iter+1;
    rx=fmin + (fmax-fmin)*rand;
    rphi=gauss(rx,m,sigma2);
    ry=rphi'*w;
    err=feval(fun,rx)-ry;
    w=w+eta*err*rphi;
    esum=esum+sqrt(sum(err.*err));
  end
  psum((iter-iterstart)/itersub)=esum;

    y=Phi*w;

    subplot(3,1,1); 
    hold on;
    plot(iterstart+1:itersub:iterstop,log(psum),color);
    xlim([0, totalIter]);
    grid on;
    hold off;
    title(['RBF-units=', int2str(units), ', eta:',num2str(eta),', ', alg, ': log(error) vs iter']);

    subplot(3,1,2); 
    plot(x,y,x,f);
    hold on;
    scatter(m, zeros(size(m,1),1),'k.');
    hold off;
    grid on;
    axis([0,7,-1.5,1.5]);
    title('Function y and desired y');

    subplot(3,1,3); 
    plot(x,f-y);
    grid on;
    title(['Residual, max= ', num2str(max(abs(f-y))), ' mean=', num2str(mean(abs(f-y)))]);
    pause(0);

end
iter=iterstop;

y=Phi*w;

subplot(3,1,1); 
hold on;
plot(iterstart+1:itersub:iterstop,log(psum),color);
xlim([0, totalIter]);
grid on;
hold off;
title(['RBF-units=', int2str(units), ', eta:',num2str(eta),', ', alg, ': log(error) vs iter']);

subplot(3,1,2); 
plot(x,y,x,f);
hold on;
scatter(m, zeros(size(m,1),1),'k.');
hold off;
grid on;
axis([0,7,-1.5,1.5]);
title('Function y and desired y');

subplot(3,1,3); 
plot(x,f-y);
grid on;
max_abs = max(abs(f-y));
max_mean = mean(abs(f-y));
title(['Residual, max= ', num2str(max_abs), ' mean=', num2str(max_mean)]);

%Find the actual output by using the calculated weight vector

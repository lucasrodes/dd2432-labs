% Do batchwise EM-steps until convergence

diffm = 1;
diffsigma2 = 1;
%plotbuffer(pd);
%while (diffsigma2>1e-10) & (diffm>1e-10)
while (diffsigma2>1e-7) & (diffm>1e-8)
  emstepb;	% A batchwise EM-step
  diffm = sum(sum((m-oldm).*(m-oldm)));
  diffsigma2 =sum((sigma2-oldsigma2).*(sigma2-oldsigma2));
  %plotrefresh(pd);
  pause(0.3);
end;
%plotnobuff(pd);

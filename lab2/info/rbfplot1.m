function rbfplot1(x,y,yd,units, m, sigma2, w)
    %Plot of x, y and desired y with residual

    %Find the actual output by using the calculated weight vector
    subplot(3,1,1);
    plot(x,y,x,yd);
    hold on;
    scatter(m, zeros(size(m,1),1),'k.');
    grid on;
    title([' Function y and desired y, RBF units=' int2str(units)], 'Interpreter', 'latex','FontSize',16);
    xlabel('$x$','Interpreter', 'latex','FontSize',16);
    h_legend = legend('Desired Function', 'Approximation Function');
    set(h_legend,'FontSize',16,'Interpreter','latex');
    
    subplot(3,1,2);
    hold on;
    t = linspace(min(x), max(x), 10000);
    color = get(0,'DefaultAxesColorOrder');
    for u = 1:units
        c_idx = mod(u, 7) + 1;
        plot(t, w(u)*normpdf(t,m(u),sigma2(u)), 'Color', color(c_idx,:));
    end
    hold on;
    scatter(m, zeros(size(m,1),1),'k.');
    grid on;
    title('RBF units activations', 'Interpreter', 'latex','FontSize',16);
    xlabel('$x$','Interpreter', 'latex','FontSize',16);
    ylabel('$\phi(x)$','Interpreter', 'latex','FontSize',16)
    
    subplot(3,1,3); 
    plot(x,yd-y);
    grid on;
    title(['Residual, max= ' num2str(max(abs(yd-y)))],'Interpreter', 'latex','FontSize',16);
end


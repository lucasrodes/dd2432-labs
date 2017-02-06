function plotdata2(h, data)
    global xmin;
    global xmax;
    global ymin;
    global ymax;
    plot(data(:,1), data(:,2), '*'), hold on,
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);
end
    
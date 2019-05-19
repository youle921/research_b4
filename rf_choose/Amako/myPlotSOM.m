
function myPlotSOM(DATA, net, figName)


width = net.width;
height = net.height;
w = net.weight;
[N,D] = size(w);

if D==2
    plot(DATA(:,1),DATA(:,2),'cy.');
elseif D==3
    scatter3(DATA(:,1),DATA(:,2),DATA(:,3),'cy.','MarkerFaceColor','cy');
end

hold on;


for i=1:N-1
    nib = getNeighborNode(i, height, width, 0);
    for j=1:numel(nib)
        plot([w(i,1) w(nib(j),1)],[w(i,2) w(nib(j),2)],'w','LineWidth',1.5);
    end
end

plot(w(:,1),w(:,2),'r.','MarkerSize',30);


axis equal
grid on
hold off
axis([0 1 0 1]);
title(figName,'fontsize',12);
pause(0.01);

end
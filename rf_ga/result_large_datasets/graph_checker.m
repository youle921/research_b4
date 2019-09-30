dataname = 'Penbased';

filepath = [dataname '_graph_data'];

close all
for i = 1 : 20
    
    data = csvread([filepath int2str(i) '.csv']);
    
    figure(i);
    semilogx(gca, 1:1001, data(:, 1));
    hold on
    semilogx(gca, 1:1001, data(:, 3));
    hold off
    
    pause
    
    close all
    
end

close all
for i = 1 : 20
    
    data = csvread([filepath int2str(i) '.csv']);
    
    figure(i)
    semilogx(1:1001, data(:, 2));
    hold on
    semilogx(1:1001, data(:, 4));
    hold off
    
    pause
    
    close all
end
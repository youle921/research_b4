filepath = 'rf_ga\result\';
list = dir([filepath  'validation*.csv']);

for i = 1 : length(list)

    disp(list(i).name)
    filename = [filepath list(i).name];
    comp = csvread(filename);

    [p, h] = ranksum(comp(:, 1), comp(:, 3));

    if h
        disp("有意水準5％で中央値が等しくない")
    else
        disp("有意水準5％で中央値が等しいという仮説が棄却できない")
    end

    disp(p)

    [p, h] = ranksum(comp(:, 2), comp(:, 3));

    if h
        disp("有意水準5％で中央値が等しくない")
    else
        disp("有意水準5％で中央値が等しいという仮説が棄却できない")
    end

    disp(p)
    disp('------------------------------------------------')
end

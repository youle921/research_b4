filepath = 'rf_ga\result_t100\';
list = dir([filepath  'oob*.csv']);

for i = 1 : length(list)

    disp(list(i).name)
    filename = [filepath list(i).name];
    comp = csvread(filename);
    
    base = comp(1:20, 1);
       
    [p, h] = ranksum(base, comp(1:20, 2));

    if h
        disp("有意水準5％で中央値が等しくない")
        if mean(base) > mean(comp(1:20, 2))
            disp("基準より低い")
        else
            disp("基準より高い")
        end
    else
        disp("有意水準5％で中央値が等しくないという仮説が棄却できない")
        disp(p)

    end
    
    pause

    disp('------------------------------------------------')

end

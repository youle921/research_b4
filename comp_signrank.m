filepath = 'Iris_10CV\';
list = dir([filepath  'result*.csv']);

method_name = ["random" "centroid" "elite"];

for i = 1 : length(list)

    disp(list(i).name)
    filename = [filepath list(i).name];
    comp = csvread(filename);
    
    base = comp(1:20, 1);

    for j = 2 : length(method_name) + 1
        
        disp(method_name(j - 1))
        [p, h] = signrank(base, comp(1:20, j));

        if h
            disp("有意水準5％で中央値が等しくない")
            if mean(base) > mean(comp(1:20, j))
                disp("基準より低い")
            else
                disp("基準より高い")
            end
        else
            disp("有意水準5％で中央値が等しくないという仮説が棄却できない")
            disp(p)

        end
    end
    disp('------------------------------------------------')
    
    if i ~= length(list)
        pause
    end
end

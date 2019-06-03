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
            disp("�L�Ӑ���5���Œ����l���������Ȃ�")
            if mean(base) > mean(comp(1:20, j))
                disp("����Ⴂ")
            else
                disp("���荂��")
            end
        else
            disp("�L�Ӑ���5���Œ����l���������Ȃ��Ƃ������������p�ł��Ȃ�")
            disp(p)

        end
    end
    disp('------------------------------------------------')
    
    if i ~= length(list)
        pause
    end
end

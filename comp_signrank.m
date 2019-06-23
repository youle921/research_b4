filepath = 'rf_ga\result_t100\';
list = dir([filepath  'oob*.csv']);

for i = 1 : length(list)

    disp(list(i).name)
    filename = [filepath list(i).name];
    comp = csvread(filename);
    
    base = comp(1:20, 1);
       
    [p, h] = ranksum(base, comp(1:20, 2));

    if h
        disp("�L�Ӑ���5���Œ����l���������Ȃ�")
        if mean(base) > mean(comp(1:20, 2))
            disp("����Ⴂ")
        else
            disp("���荂��")
        end
    else
        disp("�L�Ӑ���5���Œ����l���������Ȃ��Ƃ������������p�ł��Ȃ�")
        disp(p)

    end
    
    pause

    disp('------------------------------------------------')

end

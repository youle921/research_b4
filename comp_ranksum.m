filepath = 'rf_ga\result_v2\';
list = dir([filepath  'oob*.csv']);

for i = 1 : length(list)

    disp(list(i).name)
    filename = [filepath list(i).name];
    comp = csvread(filename);

    [p, h] = ranksum(comp(:, 1), comp(:, 3));

    if h
        disp("�L�Ӑ���5���Œ����l���������Ȃ�")
    else
        disp("�L�Ӑ���5���Œ����l���������Ȃ��Ƃ������������p�ł��Ȃ�")
    end

    disp(p)

    [p, h] = ranksum(comp(:, 2), comp(:, 3));

    if h
        disp("�L�Ӑ���5���Œ����l���������Ȃ�")
    else
        disp("�L�Ӑ���5���Œ����l���������Ȃ��Ƃ������������p�ł��Ȃ�")
    end

    disp(p)
    disp('------------------------------------------------')
end

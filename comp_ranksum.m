comp1_filename = 'result_add_pima.csv';
comp2_filename = 'result_add_pima.csv';

comp1 = csvread(comp1_filename);
comp2 = csvread(comp2_filename);

for i = 1:5
    [p, h] = ranksum(comp1((i - 1) * 200 + 1 : i * 200, 1), comp2((i - 1) * 200 + 1 : i * 200, 2));

    if h
        disp("�L�Ӑ���5���Œ����l���������Ƃ������������p")
    else
        disp("�L�Ӑ���5���Ŋ��p�ł��Ȃ�")
    end

    disp(p)
end

for i = 1:5
    [p, h] = ranksum(comp1((i - 1) * 200 + 1 : i * 200, 1), comp2((i - 1) * 200 + 1 : i * 200, 3));

    if h
        disp("�L�Ӑ���5���Œ����l���������Ƃ������������p")
    else
        disp("�L�Ӑ���5���Ŋ��p�ł��Ȃ�")
    end

    disp(p)
end

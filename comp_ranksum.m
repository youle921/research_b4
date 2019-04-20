comp1_filename = 'result_add_pima.csv';
comp2_filename = 'result_add_pima.csv';

comp1 = csvread(comp1_filename);
comp2 = csvread(comp2_filename);

for i = 1:5
    [p, h] = ranksum(comp1((i - 1) * 200 + 1 : i * 200, 1), comp2((i - 1) * 200 + 1 : i * 200, 2));

    if h
        disp("有意水準5％で中央値が等しいという仮説が棄却")
    else
        disp("有意水準5％で棄却できない")
    end

    disp(p)
end

for i = 1:5
    [p, h] = ranksum(comp1((i - 1) * 200 + 1 : i * 200, 1), comp2((i - 1) * 200 + 1 : i * 200, 3));

    if h
        disp("有意水準5％で中央値が等しいという仮説が棄却")
    else
        disp("有意水準5％で棄却できない")
    end

    disp(p)
end

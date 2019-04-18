datalist = ["Vehicle" "Pima" "vowel" "heart" "glass"];

parfor i = 1 : length(datalist)
    
    dataname = char(datalist(i));
    filename = ['..\dataset\' dataname '.csv']; 
    T = readtable(filename);
    data = T(:, 1:size(T, 2) - 1);
    answer = table2array(T(:, size(T, 2)));
    class = unique(answer);

    rng(10)
    cv = cvpartition(answer, 'KFold', 4);
    train_data = data(~cv.test(1), :);
    test_data = data(cv.test(1), :);
    train_ans = answer(~cv.test(1), :);
    test_ans = answer(cv.test(1), :);   
    
    seed = i;
    
    acc = rf_ga_framework(seed, train_data, train_ans, test_data, test_ans, class, 'oob')
    disp(dataname)
    disp({'init' acc(1)})
    disp({'obtain' acc(2)})
    disp({'origin' acc(3)})
end


mkdir result_v2

% datalist = ["Vehicle" "Pima" "Vowel" "Heart" "Glass" "Satimage"];
datalist = ["Vehicle" "Pima" "Vowel"];

for i = 1 : length(datalist)
    
    dataname = char(datalist(i));
    filename = ['..\dataset\' dataname '.csv']; 
    T = readtable(filename);
    data = T(:, 1:size(T, 2) - 1);
    answer = table2array(T(:, size(T, 2)));
    class = unique(answer);

    cv_num = 2;
    cv_div = 10;
    acc_list = zeros(cv_num * cv_div, 3);
    
    method = 'validation';
    
    for cv_count = 1 : cv_num
        rng(cv_count)
        cv = cvpartition(answer, 'KFold', cv_div);
        acc_tmp = zeros(cv_div, 3);
        
        parfor cv_trial = 1 : cv_div

            train_data = data(~cv.test(cv_trial), :);
            test_data = data(cv.test(cv_trial), :);
            train_ans = answer(~cv.test(cv_trial), :);
            test_ans = answer(cv.test(cv_trial), :);   

            seed = (cv_count - 1) * 10 + cv_trial;

            acc_tmp(cv_trial, :) = rf_ga_framework(seed, train_data, train_ans, test_data, test_ans, class, method);

        end
        
        acc_list((cv_count - 1) * 10 + 1: cv_count * 10, :) = acc_tmp;

    end
    
    csvwrite(['result_v2\' method '_' dataname '.csv'], acc_list);
    disp([dataname ' finished'])
end

disp(['----' method ' method result----'])
disp('first column is init')
disp('second column is best')
disp('third column is base')
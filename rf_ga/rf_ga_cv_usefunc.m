addpath('..\dataset')
addpath('..\rf_ga_func')

dirname = 'result_t250test';
mkdir(dirname);

datalist = ["Vehicle" "Pima" "vowel" "heart" "glass" "Satimage"];
method_list = {'oob', 'validation'};
cv_num = 2;
        
for method_no = 1 : length(method_list)
    
    method = method_list{method_no};
    
    for i = 1 : length(datalist) - 5

        dataname = char(datalist(i));
        filename = [dataname '.csv']; 
        T = readtable(filename);
        data = T(:, 1:size(T, 2) - 1);
        answer = T(:, size(T, 2));
        class = table2array(unique(answer));

        save_data = zeros(10 * cv_num, 3);

        for cv_cnt = 1 : cv_num
            rng(cv_num)
            cv = cvpartition(answer{:,1}, 'KFold', 10);

            for cv_trial = 1 : 10
                train_data = data(~cv.test(cv_trial), :);
                test_data = data(cv.test(cv_trial), :);
                train_ans = answer(~cv.test(cv_trial), :);
                test_ans = answer(cv.test(cv_trial), :);   

                c_try_num = cv_trial + 10 * (cv_cnt - 1);
                seed = c_try_num;

                params = ga_framework(seed, train_data, train_ans, test_data, test_ans, class, method);
                prd_base = rf_get_predict(params.rf_model, test_data, class);
                prd_best = rf_get_predict(params.rf_model, test_data, class, params.pop_list(1, :));

                save_data(c_try_num, 1) = sum(prd_base(:, 1) == table2array(test_ans)) / height(test_ans);
                save_data(c_try_num, 2) = sum(prd_best(:, 1) == table2array(test_ans)) / height(test_ans);
                save_data(c_try_num, 3) = sum(params.pop_list(1, :));
            end
        end

        csvwrite([dirname '\' method '_' dataname '.csv'], save_data);
        disp([dataname ' finished'])

    end
    
end
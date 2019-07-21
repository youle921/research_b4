addpath('..\dataset')
addpath('..\rf_ga_func')

oob_ratio_list = [0.6 0.5 0.4 0.3 0.2 0.1];

datalist = ["Vehicle" "Pima" "heart" "glass" "Satimage"];
cv_num = 2;

tree_num = 50;
method_params.name = 'oob';
    
for ratio_num = 1 : length(oob_ratio_list)

    dirname = ['result_choose' (int2str(oob_ratio_list(ratio_num)))];
    mkdir(dirname);
    method_params.choose_ratio = oob_ratio_list(ratio_num);

    for i = 1 : length(datalist)

        dataname = char(datalist(i));
        filename = [dataname '.csv']; 
        T = readtable(filename);
        data = T(:, 1:size(T, 2) - 1);
        answer = T(:, size(T, 2));
        class = table2array(unique(answer));

        save_data = zeros(10 * cv_num, 3);

        for cv_cnt = 1 : cv_num
            rng(cv_cnt)
            cv = cvpartition(answer{:,1}, 'KFold', 10);
            base_tmp = zeros(10, 1);
            best_tmp = zeros(10, 1);
            tree_tmp = zeros(10, 1);
            
            parfor cv_trial = 1 : 10
                train_data = data(~cv.test(cv_trial), :);
                test_data = data(cv.test(cv_trial), :);
                train_ans = answer(~cv.test(cv_trial), :);
                test_ans = answer(cv.test(cv_trial), :);   

                c_try_num = cv_trial + 10 * (cv_cnt - 1);
                seed = c_try_num;

                params = ga_framework(seed, train_data, train_ans, tree_num, method_params);
                prd_base = rf_get_predict(params.rf_model, test_data, class);
                prd_best = rf_get_predict(params.rf_model, test_data, class, params.pop_list(1, :));

                base_tmp(cv_trial) = sum(prd_base(:, 1) == table2array(test_ans)) / height(test_ans);
                best_tmp(cv_trial) = sum(prd_best(:, 1) == table2array(test_ans)) / height(test_ans);
                tree_tmp(cv_trial) = sum(params.pop_list(1, :));
            end
            
            save_data(10 * cv_cnt - 9 : cv_cnt * 10, 1) = base_tmp;
            save_data(10 * cv_cnt - 9 : cv_cnt * 10, 2) = best_tmp;
            save_data(10 * cv_cnt - 9 : cv_cnt * 10, 3) = tree_tmp;
        end

        csvwrite([dirname '\' method_params.name '_' dataname '.csv'], save_data);
        disp([dataname ' finished'])

    end
    
end
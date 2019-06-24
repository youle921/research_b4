addpath('..\dataset')
addpath('..\rf_ga_func')

t_num_list = [100 150 200 250 300 350 400 450 500];

datalist = ["Vehicle" "Pima" "heart" "glass" "Satimage"];
cv_num = 2;
method = 'oob';
    
for t_num = 1 : length(t_num_list)

    dirname = ['result_t' (int2str(t_num_list(t_num)))];
    mkdir(dirname);

    parfor i = 1 : length(datalist)

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

                params = ga_framework(seed, train_data, train_ans, test_data, test_ans, class, method, tree_num);
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
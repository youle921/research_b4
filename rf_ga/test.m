addpath('..\dataset')
addpath('..\rf_ga_func')

datalist = {'Vehicle'};
cv_num = 2;

method = {'validation'};

for m = 1 : length(method)
    
    method_params.name = method{m};
%     path = [dirname '\' method{m} '\' init_name{init}];
%     mkdir(path);

    for i = 1 : length(datalist)

        filename = [datalist{i} '.csv']; 
        T = readtable(filename);
        data = T(:, 1:size(T, 2) - 1);
        answer = T(:, size(T, 2));
        class = table2array(unique(answer));

        save_data = zeros(10 * cv_num, 3);

        for cv_cnt = 1 : cv_num

            base_tmp = zeros(10, 1);
            best_tmp = zeros(10, 1);
            tree_tmp = zeros(10, 1);

            for cv_trial = 1 : 10

                c_try_num = cv_trial + 10 * (cv_cnt - 1);
                seed = c_try_num;
                
                ga = class_randomforest_GA(data, answer);
                ga = ga.set_separator(seed);
                ga = ga.set_GA('bin', 'mutation');

                ga = ga.GA(seed, cv_trial, method_params);

                base_tmp(cv_trial) = ga.get_default_acc(cv_trial);
                best_tmp(cv_trial) = ga.get_best_acc(cv_trial);

            end

            save_data(10 * cv_cnt - 9 : cv_cnt * 10, 1) = base_tmp;
            save_data(10 * cv_cnt - 9 : cv_cnt * 10, 2) = best_tmp;
            save_data(10 * cv_cnt - 9 : cv_cnt * 10, 3) = tree_tmp;
        end

        csvwrite([path '\' datalist{i} '.csv'], save_data);
        disp([datalist{i} ' finished'])

    end

    
end
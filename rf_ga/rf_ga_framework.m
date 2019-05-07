function acc = rf_ga_framework(seed, train_data, train_ans, test_data, test_ans, class, method)

    tree_num = 50;
    p_num = 50;
    c_num = 50;
    gen = 1000;
    
    valid_data = [];
    valid_ans = [];
    
    if strcmp(method, 'validation')
        rng(seed)
        cv = cvpartition(train_ans, 'KFold', 5);
        valid_data = train_data(cv.test(1), :);
        valid_ans = train_ans(cv.test(1), :);   
        train_data = train_data(~cv.test(1), :);
        train_ans = train_ans(~cv.test(1), :);
    end
    
% 実行確認用
%     if strcmp(method, 'validation')
%         valid_data = test_data;
%         valid_ans = test_ans;   
%     end

    ga_method = class_randomforest_GA(seed, tree_num, train_data, train_ans, class, p_num, c_num, method, valid_data, valid_ans);
    
    acc = zeros(3,1);
    acc(1) = ga_method.get_best_acc(test_data, test_ans);
    acc(3) = ga_method.get_default_acc(test_data, test_ans);

    for i = 1 : gen
        ga_method = ga_method.generate_next_generation();
    end

    acc(2) = ga_method.get_best_acc(test_data, test_ans);
    
end
dataname = 'vehicle';
filename = ['..\dataset\' dataname '.csv']; 
T = readtable(filename);
data = T(:, 1:size(T, 2) - 1);
answer = table2array(T(:, size(T, 2)));
class = unique(answer);

tree_num = 100;
method_list = {'class_acc', 'prd_tsne', 'random'};
ratio_list = [0.09 0.20 0.30 0.39 0.51];

cv_num = 1;

rf = class_randomforest;
rf.class_list = class;
rf.t_num = tree_num;
acc_list = zeros(20, 5, 4);

% for rate = 1:length(ratio_list)
for rate = 2
    rf.choose_ratio = ratio_list(rate);
    origin_acc = zeros(cv_num, 1);
    chosen_acc = zeros(cv_num, 3);
    
    for i = 1:cv_num

        rng(i);
        cv = cvpartition(answer, 'KFold', 10);

        for j = 1:1
            seed = i * 10 - 10 + j;
            
            rf.train_data = data(~cv.test(j), :);
            rf.train_answer = answer(~cv.test(j));
            rf.test_data = data(cv.test(j), :);

            trees = rf.get_trees(seed);
            origin_acc(i) = origin_acc(i) + sum(rf.predict(trees, seed) == answer(cv.test(j)));
            for k = 1:3
               chosen_acc(i, k) = chosen_acc(i, k) + sum(rf.predict(trees, seed, method_list(k))...
                   == answer(cv.test(j)));
            end
        end

    end
    
    save_data = horzcat(origin_acc, chosen_acc) / length(answer)
%     csvwrite(['result_' dataname '_' int2str(rate) '.csv'], save_data);
%     acc_list(:, rate, :) = save_data;
end

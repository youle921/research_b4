% tic

tree_num = 26

addpath('..\dataset')
addpath('..\rf_ga_func')

result_dir = ['result_t' int2str(tree_num)];
mkdir(result_dir)

datalist = {'Vehicle' 'Pima' 'heart' 'glass' 'Satimage'};

for d = 1 : length(datalist)

    filename = [datalist{d} '.csv']; 
    T = readtable(filename);
    data = T(:, 1:size(T, 2) - 1);
    answer = T(:, size(T, 2));
    class = table2array(unique(answer));

    rng(10)
    cv = cvpartition(answer{:,1}, 'KFold', 4);
    train_data = data(~cv.test(1), :);
    test_data = data(cv.test(1), :);
    train_ans = answer(~cv.test(1), :);
    test_ans = answer(cv.test(1), :);   

    seed = d;

    params = ga_framework(seed, train_data, train_ans, test_data, test_ans, class, 'validation_test', tree_num);

    prd_array = cell(height(test_ans), tree_num);
    for t = 1 : params.tree_num
        prd_array(:, t) = predict(params.rf_model.Trees{t}, test_data);
    end
    prd_array = cellfun(@str2num, prd_array);

    cmb_num = 2^tree_num;
    batch_num = fix(maxNumCompThreads('automatic') / 4) * 4;
    batch_size = cmb_num / batch_num;
    acc = zeros(batch_size, batch_num);
    tree_id = logical(dec2bin(1 : cmb_num - 1, tree_num) - '0');

    div_tree_id = cell(batch_num, 1);
    for i = 1:batch_num
        if i == 1
            div_tree_id(1) = {tree_id(1 : batch_size - 1, :)};
        else
            div_tree_id(i) = {tree_id((i - 1) * batch_size : i * batch_size - 1, :)};
        end
    end

    parfor i = 1:batch_num

        if i == 1
            acc_tmp = zeros(batch_size, 1);
            acc_tmp(2 : batch_size) = aggregate_function(div_tree_id{i}, prd_array, test_ans);
            acc(:, i) = acc_tmp;
        else
            acc(:, i) = aggregate_function(div_tree_id{i}, prd_array, test_ans);
        end

    end

    acc = acc(2 : cmb_num);
    
    save_dir = [result_dir '\' datalist{d}];
    mkdir(save_dir)
    save([save_dir '\acc_list'], 'acc');
    save([save_dir '\ga_params'], 'params');

end

toc

% mean_acc = zeros(tree_num, 1);
% for t = 1 : tree_num
%     mean_acc(t) = mean(acc(sum(tree_id, 2) == t));
% end
% 
% ga_acc = params.score;
% 
% % view histogram bined each acc
% 
% histogram(acc, unique(acc));
% xlabel('Accuracy');
% ylabel('Number of Classifiers')
% 
% datetime

% compare histogram

% h1 = histogram(acc, unique(acc));
% hold on
% h2 = histogram(ga_acc, unique(acc));
% h1.Normalization = 'probability';
% h2.Normalization = 'probability';
% xlabel('Accuracy');
% ylabel('Number of Classifiers');

%% func to parallel
function acc = aggregate_function(id, prd, answer)

p_num = size(id, 1);
acc = zeros(p_num, 1);
    
answer = table2array(answer);

for i = 1:p_num
    prd_tmp = mode(prd(:, id(i, :)), 2);
    acc(i) = mean(prd_tmp == answer);
end

end

% hold on
% plot(avg_t_num)
% scatter(1:20, avg_t_num(:, 1), 35, [0 0.447 0.741], 'filled')
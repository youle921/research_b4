% tic

tree_num = 26;

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
    cv = cvpartition(answer{:,1}, 'KFold', 10);
    train_data = data(~cv.test(1), :);
    test_data = data(cv.test(1), :);
    train_ans = answer(~cv.test(1), :);
    test_ans = answer(cv.test(1), :);   

    seed = d;
    
    params.tree_num = tree_num;
    params.p_num = 50;
    params.c_num = 50;
    gen_num = 1000;

    rng(seed)
    cv = cvpartition(train_ans{:, 1}, 'KFold', 5);
    valid_data = train_data(cv.test(1), :);
    valid_ans = train_ans(cv.test(1), :);   
    train_data = train_data(~cv.test(1), :);
    train_ans = train_ans(~cv.test(1), :);

%% initialize ga

    rng(seed);

    params.rf_model = TreeBagger(params.tree_num, train_data, train_ans, ...
        'OOBPrediction', 'on');
    params.pop_list = logical(round(rand(params.p_num, params.tree_num)));

%% get predict array

    prd_array = cell(height(valid_ans), params.tree_num);
    for t = 1 : params.tree_num
        prd_array(:, t) = predict(params.rf_model.Trees{t}, valid_data);
    end
    prd_array = cellfun(@str2num, prd_array);
    score_ans = valid_ans;

    params.score = aggregate_function(params.pop_list, prd_array, score_ans);

%% generate next gen
    for gen = 1:gen_num
        [params.pop_list, params.score] = update_pop(params, prd_array, score_ans);
    end

    prd_array_test = cell(height(test_ans), tree_num);
    for t = 1 : params.tree_num
        prd_array_test(:, t) = predict(params.rf_model.Trees{t}, test_data);
    end
    prd_array_test = cellfun(@str2num, prd_array_test);

    cmb_num = 2^tree_num;
    batch_num = fix(sqrt(maxNumCompThreads('automatic'))) ^ 2;
    batch_size = cmb_num / batch_num;
    acc_valid = zeros(batch_size, batch_num);
    acc_test = zeros(batch_size, batch_num);
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
            acc_tmp(2 : batch_size) = aggregate_function(div_tree_id{i}, prd_array, valid_ans);
            acc_valid(:, i) = acc_tmp;
            
            acc_tmp(2 : batch_size) = aggregate_function(div_tree_id{i}, prd_array_test, test_ans);
            acc_test(:, i) = acc_tmp;
        else
            acc_valid(:, i) = aggregate_function(div_tree_id{i}, prd_array, valid_ans);
            acc_test(:, i) = aggregate_function(div_tree_id{i}, prd_array_test, test_ans);
        end

    end

    acc_valid = acc_valid(2 : cmb_num);
    acc_test = acc_test(2 : cmb_num);
    
    save_dir = [result_dir '\' datalist{d}];
    mkdir(save_dir)
    save([save_dir '\valid_acc_list'], 'acc_valid');
    save([save_dir '\test_acc_list'], 'acc_test');
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

%% GA functions

function [pop_list, score] = update_pop(params, prd, answer)

    children = get_children(params);

    c_score = aggregate_function(children, prd, answer);
    
    c_score(ismember(children, params.pop_list, 'rows')) = 0;
    [~, unique_id] = unique(children, 'stable', 'rows');
    c_score(~ismember(1 : params.tree_num, unique_id)) = 0;

    tmp_value = vertcat(params.score, c_score);
    tmp_pop = vertcat(params.pop_list, children);
    [~, id] = sort(tmp_value, 'descend');
    pop_list = tmp_pop(id(1 : params.p_num), :);
    score = tmp_value(id(1 : params.p_num));

end

function children = get_children(params)

    crossover_rate = 0.9;
    crossover_rand = rand(params.c_num, 1);

    first_parent = get_parent(params);
    second_parent = get_parent(params);

    choose_id = logical(round(rand(params.c_num, params.tree_num)));
    children = second_parent;
    children(choose_id) = first_parent(choose_id);
    children(crossover_rand > crossover_rate) = first_parent(crossover_rand > crossover_rate);

    children = mutation(children);           

end

function parent = get_parent(params)

    parent_id = randi(params.p_num, params.c_num, 2);

    [~, winner] = max(params.score(parent_id), [], 2);
    [~, tmp] = max(fliplr(params.score(parent_id)), [], 2);
    cnt = sum(winner == tmp);
    winner(winner == tmp) = randi(2, cnt, 1);

    chosen_parent = diag(parent_id(:, winner)); %”z—ñ‘€ì‚ª‚í‚©‚ç‚È‚¢‚Ì‚Å‚â‚Á‚Â‚¯
    parent = params.pop_list(chosen_parent, :);      

end

function new_population = mutation(population, mutation_ratio)

    if nargin < 2
        mutation_ratio = 2 / size(population, 2);
    end            

    new_population = population;
    mutation_index = rand(size(population)) < mutation_ratio;
    new_population(mutation_index) = ~population(mutation_index);

end   

%% evaluate functions
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
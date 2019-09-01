params.tree_num = 30;
params.p_num = 50;

pop = uniformly_pop(params);
s = sum(pop, 2);

function [params, acc_list] = ga_framework(seed, train_data, train_ans, tree_num, method_params, test_data, test_ans)

params.tree_num = tree_num;
params.p_num = 50;
params.c_num = 50;
gen_num = 1000;

if strcmp(method_params.name, 'validation')
    rng(seed)
    cv = cvpartition(train_ans{:, 1}, 'KFold', 9);
    valid_data = train_data(cv.test(1), :);
    valid_ans = train_ans(cv.test(1), :);   
    train_data = train_data(~cv.test(1), :);
    train_ans = train_ans(~cv.test(1), :);
    
    method_params.choose_ratio = 1.0;
    
% confirm program
elseif strcmp(method_params.name, 'validation_test')

    valid_data = test_data;
    valid_ans = test_ans;
    
    method_params.choose_ratio = 1.0;
end

%% initialize ga

rng(seed);

params.rf_model = TreeBagger(params.tree_num, train_data, train_ans, ...
    'OOBPrediction', 'on', 'InBagFraction', method_params.choose_ratio);
params.pop_list = logical(round(rand(params.p_num, params.tree_num)));

acc_list = zeros(gen_num, 4);

%% get predict array

if strcmp(method_params.name, 'validation') || strcmp(method_params.name, 'validation_test')
    prd_array = cell(height(valid_ans), params.tree_num);
    for t = 1 : params.tree_num
        prd_array(:, t) = predict(params.rf_model.Trees{t}, valid_data);
    end
    prd_array = cellfun(@str2num, prd_array);
    score_ans = valid_ans;
end

if strcmp(method_params.name, 'oob')
    prd_array = cell(height(train_ans), params.tree_num);
    for t = 1 : params.tree_num
        prd_array(:, t) = predict(params.rf_model.Trees{t}, train_data);
    end
    
    prd_array = cellfun(@str2num, prd_array);    
    prd_array(~params.rf_model.OOBIndices) = nan;
    score_ans = train_ans;
end

params.score = aggregate_function(params.pop_list, prd_array, score_ans);

%% generate next gen
for gen = 1:gen_num
    [params.pop_list, params.score] = update_pop(params, prd_array, score_ans);
end

end

function init_pop = uniformly_pop(params)

    init_pop = zeros(params.p_num, params.tree_num);
    n = fix(params.p_num / params.tree_num);
    m = mod(params.p_num, params.tree_num);
    
    current_num = 0;
    for i = 1 : n        
        for j = 1 : params.tree_num
            init_pop(current_num + j, randperm(params.tree_num, j)) = 1;
        end
        current_num = i * params.tree_num;
    end
    
    rand_id = randperm(params.tree_num, m);
    for i = 1 : m
        init_pop(current_num + i, randperm(params.tree_num, rand_id(i))) = 1;
    end
    
end

function init_pop = contain_pop(params)
    init_pop = zeros(params.p_num, params.tree_num);
    init_pop(1 : params.p_num - 2, :) = logical(round(rand(params.p_num - 2, params.tree_num)));
    init_pop(params.p_num - 1, randi(params.tree_num, 1)) = 1;
    init_pop(params.p_num, :) = 1;
end

function [pop_list, score] = update_pop(params, prd, answer)

    children = get_children(params);

    c_score = aggregate_function(children, prd, answer);
    
%%     check overlapping
    c_score(ismember(children, params.pop_list, 'rows')) = 0;
    [~, unique_id] = unique(children, 'stable', 'rows');
    c_score(~ismember(1 : params.tree_num, unique_id)) = 0;

    tmp_value = vertcat(params.score, c_score);
    tmp_pop = vertcat(params.pop_list, children);
    [~, id] = sort(tmp_value, 'descend');
    pop_list = tmp_pop(id(1 : params.p_num), :);
    score = tmp_value(id(1 : params.p_num));

end

%% GA functions
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

datalist = ["satimage" "vehicle" "pima" "wine" "vowel"];

dataname = char(datalist(1));
filename = ['..\dataset\' dataname '.csv']; 
T = readtable(filename);
data = T(:, 1:size(T, 2) - 1);
answer = table2array(T(:, size(T, 2)));
class = unique(answer);

rng(10)
cv = cvpartition(answer, 'KFold', 3);

tree_num = 100;

acc_list = zeros(10, 2);

train_data = data(~cv.test(1), :);
test_data = data(cv.test(1), :);
train_ans = answer(~cv.test(1), :);
test_ans = answer(cv.test(1), :);   

seed = 1;
Trees = TreeBagger(tree_num, train_data, train_ans, 'Method', ... 
    'classification', 'ClassName', class).Trees;

train_prd = zeros(tree_num, size(train_data, 1));

for i = 1:tree_num
    train_prd(i,:) = predict(Trees{i}, train_data);
end          

prd_Y = calc_default_tsne_2dim(1, train_prd);
choose_id = mvn_choose(prd_Y, 0.3);

origin_prd = rf_get_predict(Trees, test_data, class);
acc_list(1, 1) = sum(origin_prd(:, 1) == test_ans) / length(test_ans);
choose_prd = rf_get_predict(Trees(choose_id), test_data, class);
acc_list(1, 2) = sum(choose_prd(:, 1) == test_ans) / length(test_ans);

hold on
scatter(prd_Y(:, 1), prd_Y(:, 2), 200, [0.5 0.5 0.5], 'filled')
scatter(prd_Y(choose_id, 1), prd_Y(choose_id, 2), 200, 'r', 'filled')


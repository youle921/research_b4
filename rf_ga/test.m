datetime

tree_num = 20

addpath('..\dataset')
addpath('..\rf_ga_func')

datalist = ["Vehicle" "Pima" "vowel" "heart" "glass" "Satimage"];

i = 2;
    
dataname = char(datalist(i));
filename = [dataname '.csv']; 
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

seed = i;

params = ga_framework(seed, train_data, train_ans, test_data, test_ans, class, 'validation_test', tree_num);
mdl = params.rf_model;

cmb_num = 2^tree_num;
batch = cmb_num / 4;
acc = zeros(batch, 4);
tree_id = logical(dec2bin(1 : cmb_num - 1, tree_num) - '0');

div_tree_id = cell(4, 1);
for i = 1:4
    
    if i == 1
        div_tree_id(i) = {tree_id((i - 1) * batch + 1: i * batch - 1, :)};
    else
        div_tree_id(i) = {tree_id((i - 1) * batch : i * batch - 1, :)};
    end
    
end

parfor i = 1:4
    
    if i == 1
        acc_tmp = zeros(batch, 1);
        acc_tmp(2 : batch) = get_acc(mdl, test_data, test_ans, div_tree_id{i})
        acc(:, i) = acc_tmp;
    else
        acc(:, i) = get_acc(mdl, test_data, test_ans, div_tree_id{i});
    end
    
end

acc = acc(2 : cmb_num);

mean_acc = zeros(tree_num, 1);
for t = 1 : tree_num
    mean_acc(t) = mean(acc(sum(tree_id, 2) == t));
end

ga_acc = params.score;

% view histogram bined each acc

histogram(acc, unique(acc));
xlabel('Accuracy');
ylabel('Number of Classifier')
set(gca, 'Fontsize', 24);

datetime

% compare histogram

h1 = histogram(acc, unique(acc));
hold on
h2 = histogram(ga_acc, unique(acc));
h1.Normalization = 'probability';
h2.Normalization = 'probability';
xlabel('Accuracy');
ylabel('Probability of Classifier')
    
legend('All Classifiers', 'Obtained Classifiers')
set(gca, 'Fontsize', 24);

%% func to parallel
function acc_list = get_acc(mdl, data, answer, id)

num = size(id, 1);
acc_list = zeros(num, 1);
for i = 1:size(id, 1)
    acc_list(i) = 1 - error(mdl, data, answer, 'Mode', 'ensemble', 'Trees', find(id(i, :)));
end

end

% hold on
% plot(avg_t_num)
% scatter(1:20, avg_t_num(:, 1), 35, [0 0.447 0.741], 'filled')
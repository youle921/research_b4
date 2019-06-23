addpath('..\dataset')

datalist = ["Vehicle" "Pima" "heart" "glass" "Satimage"];

%  for i = 1 : length(datalist)

i = 5;

dataname = char(datalist(i));
filename = [dataname '.csv']; 
T = readtable(filename);
data = T(:, 1:size(T, 2) - 1);
answer = T(:, size(T, 2));
class = table2array(unique(answer));

rng(i)
cv = cvpartition(answer{:,1}, 'KFold', 10);

train_data = data(~cv.test(1), :);
test_data = data(cv.test(1), :);
train_ans = answer(~cv.test(1), :);
test_ans = answer(cv.test(1), :);   

tree_num = 50;

mdl = TreeBagger(tree_num, train_data, train_ans, 'OOBPrediction', 'on');

base_acc = 1 - error(mdl, test_data, test_ans, 'Mode', 'Ensemble');

oob_acc = zeros(tree_num, 1);

for t = 1 : tree_num
    id = mdl.OOBIndices(:, t);
    prd = cellfun(@str2num, predict(mdl.Trees{t}, train_data{id, :}));
    oob_acc(t) = mean(prd == train_ans{id, 1});
end
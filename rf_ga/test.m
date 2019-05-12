datalist = ["Vehicle" "Pima" "vowel" "heart" "glass" "Satimage"];

i = 6;
    
dataname = char(datalist(i));
filename = ['..\dataset\' dataname '.csv']; 
T = readtable(filename);
data = T(:, 1:size(T, 2) - 1);
answer = table2array(T(:, size(T, 2)));
class = unique(answer);

rng(10)
cv = cvpartition(answer, 'KFold', 4);
train_data = data(~cv.test(1), :);
test_data = data(cv.test(1), :);
train_ans = answer(~cv.test(1), :);
test_ans = answer(cv.test(1), :);   

seed = i;
method = 'validation';

tree_num = 50;
p_num = 50;
c_num = 50;
gen = 10;

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

ga_method = class_randomforest_GA(seed, tree_num, train_data, train_ans, class, p_num, c_num, method, valid_data, valid_ans);

first = ga_method.parent_value;

acc = zeros(3,1);
acc(1) = ga_method.get_best_acc(test_data, test_ans);
acc(3) = ga_method.get_default_acc(test_data, test_ans);

for i = 1 : gen
    ga_method = ga_method.generate_next_generation();
end

finish = ga_method.parent_value;

acc(2) = ga_method.get_best_acc(test_data, test_ans);
disp(dataname)
disp({'init' acc(1)})
disp({'obtain' acc(2)})
disp({'origin' acc(3)})
addpath('..\dataset')
addpath('..\rf_ga_func')

datalist = ["Vehicle" "Pima" "vowel" "heart" "glass" "Satimage"];

i = 6;
    
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
method = 'validation';

p = ga_framework(seed, train_data, train_ans, test_data, test_ans, class, method);

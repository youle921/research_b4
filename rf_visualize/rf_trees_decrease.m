datalist = ["satimage" "vehicle" "pima" "wine" "vowel"];

dataname = char(datalist(2));
filename = ['..\dataset\' dataname '.csv']; 
T = readtable(filename);
data = T(:, 1:size(T, 2) - 1);
answer = table2array(T(:, size(T, 2)));
class = unique(answer);

tree_num = 200;

cv = cvpartition(answer, 'KFold', 3);

train_data = data(~cv.test(1), :);
train_answer = answer(~cv.test(1));
test_data = data(cv.test(1), :);
test_answer = answer(cv.test(1));

Mdl = TreeBagger(tree_num, train_data, train_answer,'OOBPrediction','On',...
    'Method','classification', 'ClassNames', class, 'Options', statset('UseParallel', true));
trees = Mdl.Trees();

prd = zeros(tree_num, size(train_data, 1));

parfor i = 1:tree_num
    
    prd(i, :) = predict(trees{i}, train_data);
    
end
    
prd_Y = calc_default_tsne_2dim(1, prd);

scatter(prd_Y(:, 1), prd_Y(:, 2))

acc_list = zeros(tree_num - 1, 1);
tree_index = 1:tree_num;
correct = zeros(tree_num, 1);

for j = 1:tree_num-1
   
    loop_length = length(tree_index);
    
    parfor i = 1:loop_length
        index = tree_index;
        index (i) = [];
        correct(i, 1) = sum(rf_get_predict(trees(index), test_data, class) == test_answer);
    end
    
    [max_correct, max_id] = max(correct);

    tree_index(max_id) = [];
    acc_list(j) = max_correct; 
    correct = zeros(tree_num, 1);
    
    save_chosen_trees(j, prd_Y, tree_index)
end

plot(acc_list / sum(cv.test(1)), 'LineWidth', 2)
xticklabels(tree_num : -20 : 1)
xlabel('Number of Trees')
ylabel('Accuracy')
set(gca, 'Fontsize', 20);

csvwrite('accuracy.csv',acc_list);

function save_chosen_trees(num, data, id)

figure('visible','off');
hold on
    
scatter(data(:, 1), data(:, 2), 200, [0.5 0.5 0.5], 'filled')
scatter(data(id, 1), data(id, 2), 200, 'r', 'filled')

axis square
grid on
xticklabels({})
yticklabels({})

saveas(gcf, ['tree_plot' int2str(num)], 'png')

close all

end

 function predict_list = rf_get_predict(trees, data, class_list)
            
    data_num = size(data, 1);
    
    prd_tmp = zeros(data_num, length(trees));
    for i = 1:length(trees)
        prd_tmp(:, i) = predict(trees{i}, data);
    end

    counter = zeros(data_num, length(class_list));
    for i = 1:length(class_list)
       counter(:, i) = sum(prd_tmp == class_list(i), 2);  
    end

    [prd, max_id] = max(counter, [], 2);

    predict_list = zeros(data_num, 1);
    for i = 1:data_num
        predict_list(i, 1) = class_list(max_id(i));
        if sum(counter(i, :) == prd(i)) ~= 1
            predict_list(i, 1) = nan;
        end
    end 
    
 end
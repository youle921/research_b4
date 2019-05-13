close all

rng(1);
dataname = 'vehicle';
filename = ['..\dataset\' dataname '.csv']; 

T = readtable(filename);
data = T(:, 1:size(T, 2) - 1);
answer = table2array(T(:, size(T, 2)));

class = unique(answer);
class_num = size(class, 1);

rng(1)
cv = cvpartition(answer, 'KFold', 3);
train_data = data(~cv.test(1), :);
test_data = data(cv.test(1), :);
train_ans = answer(~cv.test(1), :);
test_ans = answer(cv.test(1), :); 

tree_num = 200;

rng(1)
Mdl = TreeBagger(tree_num, train_data, train_ans,'OOBPrediction','On','Method','classification', 'ClassNames', class);

% prd = zeros(tree_num, size(data, 1));
% oob_accuracy = zeros(tree_num,1);

% for i = 1:tree_num
%     
%     prd(i,:) = predict(Mdl.Trees{i}, data);
%     
%     oob_tmp = confusionmat(answer(Mdl.OOBIndices(:, i)), prd(i, Mdl.OOBIndices(:, i)));
%     oob_accuracy(i) = sum(diag(oob_tmp)) / sum(Mdl.OOBIndices(:, i));
%     
% end

test_size = length(test_ans);

train_prd = zeros(tree_num, size(train_data, 1));
test_accuracy = zeros(tree_num,1);

for i = 1:tree_num
    
    train_prd(i,:) = predict(Mdl.Trees{i}, train_data);
    test_prd = predict(Mdl.Trees{i}, test_data);
    
    test_accuracy(i) = sum(test_prd == test_ans) / test_size;
    
end

prd_Y = calc_default_tsne_2dim(1, train_prd);

visualize_bottom_to_up(prd_Y, test_accuracy, 1);
display_colorbar(test_accuracy);

% oobdist = squareform(pdist(Mdl.OOBIndices', 'cityblock'));
% mds_Y = mdscale(oobdist, 2);
% 
% visualize_bottom_to_up(mds_Y, oob_accuracy, 2);
% display_colorbar(oob_accuracy);
% 
% tsne_Y = calc_default_tsne_2dim(1, Mdl.OOBIndices' + 0);
% 
% visualize_bottom_to_up(tsne_Y, oob_accuracy, 3);
% display_colorbar(oob_accuracy);
% 
% figure(10)
% scatter(prd_Y(:, 1), prd_Y(:, 2), 200, 'MarkerEdgeColor', [0.1 0.1 0.4], 'MarkerFaceColor', 'b')
% 
% axis square
% grid on
% 
% xticklabels({})
% yticklabels({})

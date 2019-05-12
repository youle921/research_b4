rng(1);
dataname = 'iris';
filename = ['..\dataset\' dataname '.csv']; 

T = readtable(filename);
data = T(:, 1:size(T, 2) - 1);
answer = table2array(T(:, size(T, 2)));

class = unique(answer);
class_num = size(class, 1);
% counter = zeros(class_num, 1);
% for j = 1:class_num
%     counter(j,:) = sum(answer == class(j));
% end

tree_num = 200;
Mdl = TreeBagger(tree_num, data, answer,'OOBPrediction','On','Method','classification', 'ClassNames', class);

prd = zeros(tree_num, size(data, 1));
oob_accuracy = zeros(tree_num,1);

for i = 1:tree_num
    
    prd(i,:) = predict(Mdl.Trees{i}, data);
    
    oob_tmp = confusionmat(answer(Mdl.OOBIndices(:, i)), prd(i, Mdl.OOBIndices(:, i)));
    oob_accuracy(i) = sum(diag(oob_tmp)) / sum(Mdl.OOBIndices(:, i));
    
end

prd_Y = calc_default_tsne_2dim(1, prd);

visualize_bottom_to_up(prd_Y, oob_accuracy)
display_colorbar(oob_accuracy);

% 
% figure(10)
% scatter(prd_Y(:, 1), prd_Y(:, 2), 200, 'MarkerEdgeColor', [0.1 0.1 0.4], 'MarkerFaceColor', 'b')
% 
% axis square
% grid on
% 
% xticklabels({})
% yticklabels({})

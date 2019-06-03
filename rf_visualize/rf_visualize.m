close all
clear

dataname = 'satimage';
filename = ['..\dataset\' dataname '.csv']; 

T = readtable(filename);
data = T(:, 1:size(T, 2) - 1);
answer = table2array(T(:, size(T, 2)));

class = unique(answer);

tree_num = 200;

%% use original_data----------------------------------------------
% rng(1);
% Mdl = TreeBagger(tree_num, data, answer,'OOBPrediction','On','Method','classification', 'ClassNames', class,...
%     'Options', statset('UseParallel', true));
% 
% prd = zeros(tree_num, size(data, 1));
% oob_accuracy = zeros(tree_num,1);
% 
% for i = 1:tree_num
%     
%     prd(i,:) = predict(Mdl.Trees{i}, data);
%     
%     oob_tmp = confusionmat(answer(Mdl.OOBIndices(:, i)), prd(i, Mdl.OOBIndices(:, i)));
%     oob_accuracy(i) = sum(diag(oob_tmp)) / sum(Mdl.OOBIndices(:, i));
%     
% end
% 
% prd_Y = calc_default_tsne_2dim(1, prd);
% 
% visualize_bottom_to_up(prd_Y, oob_accuracy, 1);
% display_colorbar(oob_accuracy*100, "Accuracy[%]");
% xticklabels({})
% yticklabels({})
% 
% oobdist = squareform(pdist(Mdl.OOBIndices' + 0, 'cityblock'));
% mds_Y = mdscale(oobdist, 2);
% 
% visualize_bottom_to_up(mds_Y, oob_accuracy, 2);
% display_colorbar(oob_accuracy*100, "Accuracy[%]");
% xticklabels({})
% yticklabels({})
% 
% visualize_up_to_Npercent(mds_Y, oob_accuracy, 0.1, 3);
% xticklabels({})
% yticklabels({})

%% use original_data----------------------------------------------

%% use validation_data--------------------------------------------

rng(1)
cv = cvpartition(answer, 'KFold', 3);
train_data = data(~cv.test(1), :);
test_data = data(cv.test(1), :);
train_ans = answer(~cv.test(1), :);
test_ans = answer(cv.test(1), :); 

rng(1)
Mdl = TreeBagger(tree_num, train_data, train_ans,'OOBPrediction','On','Method','classification', 'ClassNames', class,...
    'Options', statset('UseParallel', true));
test_size = length(test_ans);

train_prd = zeros(tree_num, size(train_data, 1));
test_accuracy = zeros(tree_num,1);

for i = 1:tree_num
    
    train_prd(i,:) = predict(Mdl.Trees{i}, train_data);
    test_prd = predict(Mdl.Trees{i}, test_data);
    
    test_accuracy(i) = sum(test_prd == test_ans) / test_size;
    
end

prd_Y = calc_default_tsne_2dim(2, train_prd);

visualize_bottom_to_up(prd_Y, test_accuracy, 1);
display_colorbar(test_accuracy*100, "Accuracy[Åì]");
xticklabels({})
yticklabels({})

oobdist = squareform(pdist(Mdl.OOBIndices', 'cityblock'));
mds_Y = mdscale(oobdist, 2);

visualize_bottom_to_up(mds_Y, test_accuracy, 2);
display_colorbar(test_accuracy*100, "Accuracy[Åì]");
xticklabels({})
yticklabels({})

visualize_up_to_Npercent(mds_Y, test_accuracy, 0.1, 3);
xticklabels({})
yticklabels({})

%% use validation_data------------------------------------------------------

figure()
depth = get_rf_trees_depth(Mdl.Trees);
scatter3(prd_Y(:, 1), prd_Y(:, 2), depth, 200, get_colortable(test_accuracy), 'filled')
set(gca, 'Fontsize', 24);
xticklabels({})
yticklabels({})
zlabel("# of Depth")

nodes = zeros(tree_num, 1);
for i = 1 : tree_num
    nodes(i) = Mdl.Trees{i}.NumNodes;
end
figure()
scatter3(prd_Y(:, 1), prd_Y(:, 2), nodes, 200, get_colortable(test_accuracy), 'filled')
set(gca, 'Fontsize', 24);
xticklabels({})
yticklabels({})
zlabel("# of Nodes")

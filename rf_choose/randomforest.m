rng(1);
dataname = 'iris';
filename = ['..\research_b4\dataset\' dataname '.csv']; 

T = readtable(filename);
data = T(:, 1:size(T, 2) - 1);
answer = table2array(T(:, size(T, 2)));

class = unique(answer);
class_num = size(class, 1);
counter = zeros(class_num, 1);
for j = 1:class_num
    counter(j,:) = sum(answer == class(j));
end

tree_num = 200;
Mdl = TreeBagger(tree_num, data, answer,'OOBPrediction','On','Method','classification', 'ClassNames', class);

% figure();
% oobErrorBaggedEnsemble = oobError(Mdl);
% plot(oobErrorBaggedEnsemble)

prd = zeros(tree_num, size(data, 1));
% error = zeros(tree_num, class_num);
% c_matrix = zeros(tree_num, class_num^2);
% predictor = zeros(tree_num, size(data, 2));
% predictor_names = Mdl.PredictorNames;
oob_c_matrix = zeros(tree_num, class_num^2);
oob_correct = zeros(tree_num,1);

for i = 1:tree_num
    
    prd(i,:) = predict(Mdl.Trees{i}, data);
    
%     oob_counter = zeros(class_num, 1);
%     for j = 1:class_num
%         oob_counter(j,:) = max(sum(answer(Mdl.OOBIndices(:, i)) == class(j)), 1);
%     end
   
%     tmp = confusionmat(answer, prd(i, :)) ./ counter;
%     c_matrix(i, :) = reshape(tmp, 1, []);
%     error(i, :) = diag(tmp);
    oob_tmp = confusionmat(answer(Mdl.OOBIndices(:, i)), prd(i, Mdl.OOBIndices(:, i)));
%     oob_c_matrix(i, :) = reshape((oob_tmp ./oob_counter), 1, []);
    oob_correct(i) = sum(diag(oob_tmp)) / sum(Mdl.OOBIndices(:, i));

%     for k = 1:size(predictor_names,2)
%        predictor(i,k) = sum(strcmp(Mdl.Trees{i}.CutPredictor, predictor_names(k))); 
%     end
    
end

prd_Y = calc_default_tsne_2dim(1, prd);
% oob_cmat_Y = calc_default_tsne_2dim(1, oob_c_matrix);

colortable = get_colortable(oob_correct);
% colortables = zeros(tree_num, 3, class_num);
% for i = 1:class_num
%     colortables(:, :, i) = get_colortable(oob_c_matrix(:, (i + (i - 1) * class_num)));
% end

% mkdir(['â¬éãâª\' dataname])
% draw_default_scatter(10, prd_Y, colortable);
% print(['â¬éãâª\' dataname '\' int2str(tree_num) '_prd_oob'], '-dmeta')
% draw_default_scatter(20, oob_cmat_Y, colortable);
% print(['â¬éãâª\' dataname '\' int2str(tree_num) '_cmat_oob'], '-dmeta')

% for i = 1:class_num
%     draw_default_scatter(i, prd_Y, get_colortable(oob_c_matrix(:, (i + (i - 1) * class_num))));
% %     print(['â¬éãâª\' dataname '\' int2str(tree_num) '_prd_c' int2str(i)], '-dmeta')
% end
% 
% for i = 1:class_num
%     draw_default_scatter(i+3, oob_cmat_Y, get_colortable(oob_c_matrix(:, (i + (i - 1) * class_num))));
% %     print(['â¬éãâª\' dataname '\' int2str(tree_num) '_cmat_c' int2str(i)], '-dmeta')
% end

table = [0:1000] *0.001;
table = [table; table; table].';
colors = get_colortable(table);
[~, oob_id] = sort(oob_correct);
figure('Colormap', colors);
hold on
for i = 1:length(oob_id)
    id = oob_id(i);
    scatter(prd_Y(id, 1), prd_Y(id, 2), 200, 'MarkerEdgeColor', max(colortable(id, :) - 0.55, 0), 'MarkerFaceColor', colortable(id, :))
end
c = colorbar;
c.Label.String = "Accuracy[Åì]";
caxis([min(oob_correct)*100 max(oob_correct)*100])

axis square
grid on

xticklabels({})
yticklabels({})

set(gca, 'Fontsize', 24);

figure(10)
scatter(prd_Y(:, 1), prd_Y(:, 2), 200, 'MarkerEdgeColor', [0.1 0.1 0.4], 'MarkerFaceColor', 'b')

axis square
grid on

xticklabels({})
yticklabels({})

set(gca, 'Fontsize', 24);
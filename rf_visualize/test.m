datalist = ["Vehicle" "Pima" "vowel" "heart" "glass" "Satimage"];
for i = 1:5
    dataname = char(datalist(i));
    load(['results_0603\origin_data\params_'  dataname '.mat'])
    filename = [dataname '.csv']; 
    T = readtable(filename);
    data = T(:, 1:size(T, 2) - 1);
    answer = T(:, size(T, 2));
    class = unique(answer);

    rng(10)
    cv = cvpartition(answer{:,1}, 'KFold', 4);
    train_data = data(~cv.test(1), :);
    test_data = data(cv.test(1), :);
    train_ans = answer(~cv.test(1), :);
    test_ans = answer(cv.test(1), :);

    rf_predict_list = cell(200, height(train_ans));
    for t = 1 : 200
        rf_predict_list(t, :) = predict(params.rf_model.Trees{t}, train_data);
    end

    acc = 1 - error(params.rf_model, test_data, test_ans, 'Mode', 'individual');

    rf_predict_list = cellfun(@str2num, rf_predict_list);
    prd_Y = calc_default_tsne_2dim(1, rf_predict_list);
    score = sum(params.pop_list).';

%     visualize_bottom_to_up(prd_Y, score, i);
%     display_colorbar(score, "Number of chosen");
%     xticklabels({})
%     yticklabels({})
%     saveas(gcf, ['results_0603\origin_data\' dataname '.fig'])
% 
%     visualize_bottom_to_up(prd_Y, acc, i*10);
%     display_colorbar(acc*100, "Accuracy[Åì]");
%     xticklabels({})
%     yticklabels({})
%     saveas(gcf, ['results_0603\origin_data\' dataname '_acc.fig'])

    id = unique(score);
    avg_acc = zeros(length(id), 1);
    num = zeros(length(id), 1);
    for j = 1:length(id)
        avg_acc(j) = mean(acc(score == id(j))); 
        num(j) = sum(score == id(j));
    end
    graph_data = [id avg_acc];
    figure(i*100)
    plot(graph_data(:, 1), graph_data(:, 2), '-o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b')
    figure(i*100 + 1)
    plot(graph_data(:, 1), num, '-o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b')
    
end
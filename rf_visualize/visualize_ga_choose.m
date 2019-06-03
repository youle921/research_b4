addpath('..\dataset')
addpath('..\rf_ga_func')

datalist = ["Vehicle" "Pima" "vowel" "heart" "glass" "Satimage"];

for i = 1 : length(datalist)

    dataname = char(datalist(i));
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

    seed = i;
    method = 'validation';

    [rf, params] = ga_framework(seed, train_data, train_ans, test_data, test_ans, class, method);
    
    rf_predict_list = cell(200, height(train_ans));
    for t = 1 : 200
        rf_predict_list(t, :) = predict(rf.Trees{t}, train_data);
    end
    
    rf_predict_list = cellfun(@str2num, rf_predict_list);
    prd_Y = calc_default_tsne_2dim(1, rf_predict_list);
    score = sum(params.pop_list).';
    
%%     save figure
    figure('visible','off');
    visualize_bottom_to_up(prd_Y, score, i);
    display_colorbar(score, "Number of chosen");
    xticklabels({})
    yticklabels({})
   
    saveas(gcf, ['ga_chosen_' dataname], 'meta')
    close all

end


 function predict_list = rf_get_predict(mdl, data, class_list, weight)
            
    class_list = table2array(class_list);
    
    if nargin < 4
        [~, prd_tmp] = predict(mdl, data);
    else
        choose_id = find(weight);
        [~, prd_tmp] = predict(mdl, data, 'Trees', choose_id);
    end
    
    [max_num, prd] = max(prd_tmp, [], 2);

    predict_list = zeros(size(data, 1), 2);
    predict_list(:, 1) = class_list(prd);
    same_id = sum(max_num == prd_tmp, 2) > 1;
    predict_list(same_id, 1) = nan;
    predict_list(same_id, 2) = 1;
    
 end

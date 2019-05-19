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

    predict_list = zeros(data_num, 2);
    for i = 1:data_num
        predict_list(i, 1) = class_list(max_id(i));
        if sum(counter(i, :) == prd(i)) ~= 1
            predict_list(i, 1) = nan;
            predict_list(i, 2) = 1;
        end
    end 
    
 end

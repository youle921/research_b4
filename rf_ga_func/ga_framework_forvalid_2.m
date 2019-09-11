function ga_params = ga_framework_forvalid(seed, data_bundle, path, method_params)

    ga_params = set_ga_params();
    
    %% setting data for validation
    
    if strcmp(method_params.name, 'validation')
        rng(seed)
        cv = cvpartition(data_bundle.train_ans{:, 1}, 'KFold', 9);
        data_bundle.valid_data = data_uandle.train_data(cv.test(1), :);
        data_bundle.valid_ans = data_bundle.train_ans(cv.test(1), :);   
        data_bundle.train_data = data_bundle.train_data(~cv.test(1), :);
        data_bundle.train_ans = data_bundle.train_ans(~cv.test(1), :);

        method_params.choose_ratio = 1.0;
    end
    
    %% initialize GA
    
    rng(seed);

    ga_params.rf_model = TreeBagger(ga_params.tree_num, data_bundle.train_data, data_bundle.train_ans, ...
        'OOBPrediction', 'on', 'InBagFraction', method_params.choose_ratio);
    
    ga_params.pop_list = init_population(ga_params, method_params.init);
    

end

function pop = init_population(params, no)

    pop = zeros(params.p_num, params.tree_num);
    
    switch no
        %% uniform initialize
        case 1

            n = fix(params.p_num / params.tree_num);
            m = mod(params.p_num, params.tree_num);

            current_num = 0;
            for i = 1 : n        
                for j = 1 : params.tree_num
                    pop(current_num + j, randperm(params.tree_num, j)) = 1;
                end
                current_num = i * params.tree_num;
            end

            rand_id = randperm(params.tree_num, m);
            for i = 1 : m
                pop(current_num + i, randperm(params.tree_num, rand_id(i))) = 1;
            end
            
            pop = logical(pop);
            
        %% include initialization
        case 2

            pop(1 : params.p_num - 2, :) = logical(round(rand(params.p_num - 2, params.tree_num)));
            pop(params.p_num - 1, randi(params.tree_num, 1)) = 1;
            pop(params.p_num, :) = 1;
            
        %% random initialization
        otherwise
            
            pop = logical(round(rand(params.p_num, params.tree_num)));
               
    end
        
end 

function prd = get_prd_array()

    if strcmp(method_params.name, 'validation') || strcmp(method_params.name, 'validation_test')
        prd_array = cell(height(valid_ans), params.tree_num);
        for t = 1 : params.tree_num
            prd_array(:, t) = predict(params.rf_model.Trees{t}, valid_data);
        end
        prd_array = cellfun(@str2num, prd_array);
        score_ans = valid_ans;

    %     test_prd_array = cell(height(test_ans), params.tree_num);
    %     for t = 1 : params.tree_num
    %         test_prd_array(:, t) = predict(params.rf_model.Trees{t}, test_data);
    %     end
    %     test_prd_array = cellfun(@str2num, test_prd_array);

    end

    if strcmp(method_params.name, 'oob')
        prd_array = cell(height(train_ans), params.tree_num);
        for t = 1 : params.tree_num
            prd_array(:, t) = predict(params.rf_model.Trees{t}, train_data);
        end

        prd_array = cellfun(@str2num, prd_array);    
        prd_array(~params.rf_model.OOBIndices) = nan;
        score_ans = train_ans;
    end


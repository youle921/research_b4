classdef class_randomforest_GA
    properties

        data
        ans
%         validation_data
%         validation_ans
        class_num
        sep
        
        init_function
        
        params
        
        weight_list
        children_size
        population_size
        
        population_list
        parent_value
        evaluate_method
        
        train_predict
        validation_predict
        oob_predict
        
    end
   
    methods
        
        function obj = class_randomforest_GA(data, answer)
            
            obj.data = data;
            obj.ans = answer;
            obj.class_num = length(unique(obj.ans));
            
        end
        
        function obj = set_separator(obj, seed)
            
            rng(seed);
            obj.sep = cvpartition(obj.ans{:, 1}, 'KFold', 10);
            
        end
        
        function obj = set_GA(obj, code, strategy)
            
            if strcmp(code, 'bin')
                obj.init_function = @init_binary;
                
                if strcmp(strategy, 'mutation')
                    obj.updata_function = @mutation;
                else
                    obj.updata_function = @UXwithMutation;
                
            else
                obj.init_function = @init_real;
                obj.updata_function = @SBXwithPM;
            end
            
        end

        function obj = GA(obj, seed, no, method_params)
            
            train_data = obj.data(~obj.sep(no), :);
            train_ans = obj.data(obj.sep(no), :);
            
            if strcmp(method_params.name, 'validation')
                rng(seed)
                cv = cvpartition(obj.ans{~obj.sep.test(no), 1}, 'KFold', 9);
                valid_data = train_data(cv.test(1), :);
                valid_ans = train_ans(cv.test(1), :);   
                train_data = train_data(~cv.test(1), :);
                train_ans = train_ans(~cv.test(1), :);

                method_params.choose_ratio = 1.0;
            
            end

            rng(seed);

            obj.params.rf_model = TreeBagger(obj.params.t_num, obj.data(obj.sep.test(no),:), obj.train_ans, 'OOBPrediction', 'on', ...
                  'Method','classification', 'ClassName', obj.class_list);
            obj.trees = obj.rf_model.Trees;
            
            obj.population_list = logical(round(rand(obj.population_size, obj.t_num)));
            
            if strcmp(evaluate_method, 'weight')
                obj.evaluate_method = @obj.evaluation;
            end
            
            if strcmp(evaluate_method, 'oob')
                obj.evaluate_method = @obj.oob_evaluation;
            end
            
            if strcmp(evaluate_method, 'validation')
%                 obj.validation_data = valid_data;
%                 obj.validation_ans = valid_ans;
                obj.evaluate_method = @obj.validation_evaluation;
                obj.validation_predict = get_predict(obj.trees, valid_data, valid_ans, obj.class_list);
            end

            obj.parent_value = obj.evaluate_method(obj.population_list);
            
        end
        
        function acc = evaluation(obj, trees_id)
            
           p_size = size(trees_id, 1);
           acc = zeros(p_size, 1);
           for i = 1 : p_size
               prd = get_predict(obj.trees(trees_id(i, :)), obj.train_data, obj.class_list);
               acc(i) = -obj.weight_list(1) * sum(trees_id(i, :)) + obj.weight_list(2) ...
                   * sum(prd(:, 1) == obj.train_ans) / length(obj.train_ans);
           end
           
        end
        
        function acc = validation_evaluation(obj, trees_id)
            
           p_size = size(trees_id, 1);
           acc = zeros(p_size, 1);
           for i = 1 : p_size
               prd = get_predict(obj.trees(trees_id(i, :)), obj.validation_data, obj.class_list);
               acc(i) = sum(prd(:, 1) == obj.validation_ans) / length(obj.validation_ans);
           end
           
        end
        
        function acc = oob_evaluation(obj, trees_id)

            p_size = size(trees_id, 1);
            acc = zeros(p_size, 1);
            for i = 1 : p_size
                prd = get_oob_predict(obj.trees(trees_id(i, :)), obj.train_data, obj.class_list, obj.rf_model.OOBIndices(:, trees_id(i, :)));
                acc(i) = sum(prd(:, 1) == obj.train_ans) / length(obj.train_ans);
            end
           
        end
        
        function obj = generate_next_generation(obj)
            
            children = obj.get_children();
            
            acc = obj.evaluate_method(children);
            acc(ismember(children, obj.population_list, 'rows')) = 0;
            
            tmp_value = vertcat(obj.parent_value, acc);
            tmp_pop = vertcat(obj.population_list, children);
            [~, id] = sort(tmp_value, 'descend');
            obj.population_list = tmp_pop(id(1 : obj.population_size), :);
            obj.parent_value = tmp_value(id(1 : obj.population_size));
            
        end
        
%         tournament selection
%         return chosen logical index
        function parent = get_parent(obj)
            
            parent_id = randi(obj.population_size, obj.children_size, 2);
            %% select winner
            [~, winner] = max(obj.parent_value(parent_id), [], 2);
            
            %% search same value parent
            [~, tmp] = max(fliplr(obj.parent_value(parent_id)), [], 2);
            draw_cnt = sum(winner == tmp);
            winner(winner == tmp) = randi(2, draw_cnt, 1);
            
            chosen_parent = diag(parent_id(:, winner)); %îzóÒëÄçÏÇ™ÇÌÇ©ÇÁÇ»Ç¢ÇÃÇ≈Ç‚Ç¡Ç¬ÇØ
            parent = obj.population_list(chosen_parent, :);      
            
        end      
        
        function acc = get_best_acc(obj, data, answer)
            [~, best_index] = max(obj.parent_value);
            obtained_tree = obj.trees(obj.population_list(best_index, :));

            tmp = get_predict(obtained_tree, data, obj.class_list);
            acc = sum(tmp(:, 1) == answer) / length(answer);
        end
        
        function acc = get_default_acc(obj, data, answer)
            
            tmp = get_predict(obj.trees, data, obj.class_list);
            acc = sum(tmp(:, 1) == answer) / length(answer);
        end
        
        function prd_array = get_prd_array(obj, data)
            
            prd_array = zeros(size(data, 1), obj.class_num, obj.params.tree_num);
            for t = 1 : obj.params.tree_num
                [~, prd_array(:, :, t)] = predict(obj.params.rf_model.Trees{t}, data);
            end
            
        end
        
        function prd_array = get_oob_prd_array(obj, data)

            prd_array = get_prd_array(data);
            for t = 1 : obj.params.tree_num
                oob_array = repmat(obj.params.rf_model.OOBIndices(:, t), 1, obj.class_num);
                prd_array(:, :, t) = prd_array(:, :, t) .* oob_array;
            end
            
        end

        function prd = aggrigate_prediction(obj, prd_array, weight)
            
            data_num = size(prd_array, 1);
            weight = reshape(repelem(weight, data_num, obj.class_num), data_num, obj.class_num, []);
            prd = max(sum(prd_array .* weight, 3), [], 2);
            
        end
                
    end
    
end

 function predict_list = get_predict(trees, data, class_list)
            
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

 function predict_list = get_oob_predict(trees, data, class_list, oob_id)
            
    data_num = size(data, 1);
    
    prd_tmp = zeros(data_num, length(trees));
    
    for i = 1:length(trees)
        prd_tmp(oob_id(:, i), i) = predict(trees{i}, data(oob_id(:, i), :));
        prd_tmp(~oob_id(:, i), i) = nan;
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
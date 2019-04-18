classdef class_randomforest_GA
    properties

        train_data
        train_ans
        validation_data
        validation_ans
        class_list
        
        rf_model
        t_num
        trees
        
        weight_list
        children_size
        population_size
        
        population_list
        parent_value
        evaluate_method
        
    end
   
    methods
       
        function obj = class_randomforest_GA(seed, tree_num, train_data, train_ans, class_list, p_size, c_size, evaluate_method, valid_data, valid_ans)
           
            obj.t_num = tree_num;

            obj.train_data = train_data;
            obj.train_ans = train_ans; 
            obj.class_list = class_list;
            
            obj.population_size = p_size;
            obj.children_size = c_size;
            
            rng(seed);

            obj.rf_model = TreeBagger(obj.t_num, obj.train_data, obj.train_ans, 'OOBPrediction', 'on', ...
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
                obj.validation_data = valid_data;
                obj.validation_ans = valid_ans;
                obj.evaluate_method = @obj.validation_evaluation;
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
            tmp_value = vertcat(obj.parent_value, acc);
            tmp_pop = vertcat(obj.population_list, children);
            [~, id] = sort(tmp_value, 'descend');
            obj.population_list = tmp_pop(id(1 : obj.population_size), :);
            obj.parent_value = tmp_value(id(1 : obj.population_size));
            
        end

%         return logical index
        function children = get_children(obj)
            
            first_parent = obj.get_parent();
            second_parent = obj.get_parent();
            
            choose_id = logical(round(rand(obj.children_size, obj.t_num)));
            children = second_parent;
            children(choose_id) = first_parent(choose_id);
            
            children = obj.mutation(children);           
            
        end
        
%         tournament selection
%         return chosen logical index
        function parent = get_parent(obj)
            
            parent_id = randi(obj.population_size, obj.children_size, 2);
            [~, winner] = max(obj.parent_value(parent_id), [], 2);
            chosen_parent = diag(parent_id(:, winner)); %îzóÒëÄçÏÇ™ÇÌÇ©ÇÁÇ»Ç¢ÇÃÇ≈Ç‚Ç¡Ç¬ÇØ
            parent = obj.population_list(chosen_parent, :);      
            
        end
        
        function new_population = mutation(obj, population, mutation_ratio)
           
            if nargin < 3
                mutation_ratio = 1 / obj.t_num;
            end            

            new_population = population;
            mutation_index = rand(obj.children_size, obj.t_num) < mutation_ratio;
            new_population(mutation_index) = ~population(mutation_index);
            
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
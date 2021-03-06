classdef class_randomforest_GA
    properties

        data
        ans
        class_num
        sep
        
        init_function
        update_function
        evaluate_function
        
        params
 
        population_list
        parent_value
        
        score_curve
        
    end
   
    methods
        
        function obj = class_randomforest_GA(data, answer)
            
            obj.data = data;
            obj.ans = answer;
            obj.class_num = height(unique(obj.ans));
            
        end
        
        function obj = set_separator(obj, seed)
            
            rng(seed);
            obj.sep = cvpartition(obj.ans{:, 1}, 'KFold', 10);
            
        end
        
        function obj = set_GA(obj, code, strategy)
            
            if strcmp(code, 'bin')
                obj.init_function = @init_bin;
                
                if strcmp(strategy, 'mutation')
                    obj.update_function = @BitFlip;
                    obj.params = set_ga_params('p_num', 1);
                else
                    obj.update_function = @UXwithBitFlip;            
                    obj.params = set_ga_params();
                end
                
            elseif strcmp(code, 'real')
                obj.init_function = @init_real;
                obj.update_function = @SBXwithPM;
                obj.params = set_ga_params();
            end
            
        end

        function obj = GA(obj, seed, no, method_params)
            
            train_data = obj.data(~obj.sep.test(no), :);
            train_ans = obj.ans(~obj.sep.test(no), :);
            
            obj.score_curve = zeros(obj.params.gen_num + 1, 1);
            
            score_ans = train_ans;
            
            if strcmp(method_params.name, 'validation')
                
                rng(seed)
                cv = cvpartition(obj.ans{~obj.sep.test(no), 1}, 'KFold', 9);
                valid_data = train_data(cv.test(1), :);
                valid_ans = train_ans(cv.test(1), :);   
                train_data = train_data(~cv.test(1), :);
                train_ans = train_ans(~cv.test(1), :);
                
                score_ans = valid_ans;
            
            end

            rng(seed);

            obj.params.rf_model = TreeBagger(obj.params.t_num, train_data, train_ans, 'OOBPrediction', 'on', ...
                  'Method','classification');
            obj.params.trees = obj.params.rf_model.Trees;
            if strcmp(method_params.name, 'validation')
                prd_array = obj.get_prd_array(valid_data);
            elseif strcmp(method_params.name, 'oob')
                prd_array = obj.get_oob_prd_array(train_data);
            end
            
            obj.population_list = obj.init_function(obj.params.p_num, obj.params.t_num);
            obj.parent_value = obj.get_score(prd_array, obj.population_list, score_ans);
            
            obj.score_curve(1) = max(obj.parent_value);
            
            for i =1 : obj.params.gen_num
                obj = obj.generate_next_generation(prd_array, score_ans);
                obj.score_curve(i + 1) = max(obj.parent_value);
            end
            
        end
        
        function obj = generate_next_generation(obj, prd_array, score_ans)
            
            p1 = obj.tournament_selection();
            p2 = obj.tournament_selection();
            children = obj.update_function(p1, p2);
            
            acc = obj.get_score(prd_array, children, score_ans);
            acc(ismember(children, obj.population_list, 'rows')) = 0;
            
            tmp_value = horzcat(obj.parent_value, acc);
            tmp_pop = vertcat(obj.population_list, children);
            [~, id] = sort(tmp_value, 'descend');
            obj.population_list = tmp_pop(id(1 : obj.params.p_num), :);
            obj.parent_value = tmp_value(id(1 : obj.params.p_num));
            
        end

%         return chosen logical index
        function parent = tournament_selection(obj)
            
            parent_id = randi(obj.params.p_num, obj.params.c_num, 2);
            %% select winner
            [~, winner] = max(obj.parent_value(parent_id), [], 2);
            
            %% search same value parent
            [~, tmp] = max(fliplr(obj.parent_value(parent_id)), [], 2);
            draw_cnt = sum(winner == tmp);
            winner(winner == tmp) = randi(2, draw_cnt, 1);
            
            chosen_parent = diag(parent_id(:, winner)); %配列操作がわからないのでやっつけ
            parent = obj.population_list(chosen_parent, :);      
            
        end      

        function acc = get_default_acc(obj, no)
            
            test_data = obj.data(obj.sep.test(no), :);
            test_ans = obj.ans(obj.sep.test(no), :);
            
            test_prd_array = zeros(size(test_data, 1), obj.class_num, obj.params.t_num);
            for t = 1 : obj.params.t_num
                [~, test_prd_array(:, :, t)] = predict(obj.params.trees{t}, test_data);
            end
            
            w = ones(1, obj.params.t_num);
            acc = obj.get_score(test_prd_array, w, test_ans);
            
        end
        
        function acc = get_best_acc(obj, no)
            
            test_data = obj.data(obj.sep.test(no), :);
            test_ans = obj.ans(obj.sep.test(no), :);
            
            test_prd_array = zeros(size(test_data, 1), obj.class_num, obj.params.t_num);
            for t = 1 : obj.params.t_num
                [~, test_prd_array(:, :, t)] = predict(obj.params.trees{t}, test_data);
            end
            
            w = obj.population_list(1, :);
            acc = obj.get_score(test_prd_array, w, test_ans);
            
        end
        
        function prd_array = get_prd_array(obj, data)
            
            prd_array = zeros(size(data, 1), obj.class_num, obj.params.t_num);
            for t = 1 : obj.params.t_num
                [~, prd_array(:, :, t)] = predict(obj.params.trees{t}, data);
            end
            
        end
        
        function prd_array = get_oob_prd_array(obj, data)

            prd_array = obj.get_prd_array(data);
            for t = 1 : obj.params.t_num
                oob_array = repmat(obj.params.rf_model.OOBIndices(:, t), 1, obj.class_num);
                prd_array(:, :, t) = prd_array(:, :, t) .* oob_array;
            end
            
        end

        function prd = aggrigate_prediction(obj, prd_array, weight)
            
            data_num = size(prd_array, 1);
            prd = zeros(data_num, size(weight, 1));
            for p = 1 : size(weight, 1)
                w = reshape(repelem(weight(p, :), data_num, obj.class_num), data_num, obj.class_num, []);
                [~, prd(:, p)] = max(sum(prd_array .* w, 3), [], 2);
            end
            
        end
        
        function score = get_score(obj, prd_array, weight, answer)
            
            prd = obj.aggrigate_prediction(prd_array, weight);
            score = sum(prd == answer{:, :}) / height(answer);
        end
                
    end
    
end

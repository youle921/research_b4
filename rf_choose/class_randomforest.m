classdef class_randomforest
    properties
        t_num
        
        choose_ratio
        
        class_list
        train_data
        train_answer
        test_data
    end
    
    methods
        
        function trees = get_trees(obj, seed)
            rng(seed);
            trees = TreeBagger(obj.t_num, obj.train_data, obj.train_answer,...
                  'Method','classification', 'ClassName', obj.class_list).Trees;
        end
        
        function choose_id = get_choose_id(obj, seed, trees, method)

            choose_num = round(obj.t_num * obj.choose_ratio);

            if strcmp(method, 'random')
                rng(seed);
                choose_id = randsample(obj.t_num, choose_num);
            end
            
            if strcmp(method, 'prd_tsne')
                prd_list = zeros(obj.t_num, size(obj.train_data, 1));
                for i = 1:obj.t_num
                    prd_list(i, :) = predict(trees{i}, obj.train_data);
                end
                
                prd_Y = calc_default_tsne_2dim(1, prd_list);
                center = mean(prd_Y);
                choose_id = knnsearch(prd_Y, center, 'K', choose_num);
                 
                dlmwrite('prd_Y3.txt', prd_Y, '-append')
                
            end
            
            if strcmp(method, 'class_acc')
                acc_list = zeros(obj.t_num, length(obj.class_list));
                for i = 1:obj.t_num
                    prd = predict(trees{i}, obj.train_data);
                    acc_list(i, :) = diag(confusionmat(obj.train_answer, prd));
                end
                
                choose_num = max(round(choose_num / length(obj.class_list)), 1);
                [~, id] = sort(acc_list);
                choose_id = reshape(id(1:choose_num, :), [], 1);   
            end
            
            dlmwrite('choose_id3.txt', choose_id, '-append', 'roffset', 1)
        end
        
        function predict = predict(obj, trees, seed, choose_method)
            
            switch nargin
                case 4  
                    id = obj.get_choose_id(seed, trees, choose_method);
                    trees = trees(id);
            end
            
            predict = obj.get_predict(trees);

        end
        
        function predict_list = get_predict(obj, trees)
            
            prd_tmp = zeros(size(obj.test_data, 1), length(trees));
            for i = 1:length(trees)
                prd_tmp(:, i) = predict(trees{i}, obj.test_data);
            end

            counter = zeros(size(obj.test_data, 1), length(obj.class_list));
            for i = 1:length(obj.class_list)
               counter(:, i) = sum(prd_tmp == obj.class_list(i), 2);  
            end

            [prd, max_id] = max(counter, [], 2);

            predict_list = zeros(size(obj.test_data, 1), 2);
            for i = 1:size(obj.test_data, 1)
                predict_list(i, 1) = obj.class_list(max_id(i));
                if sum(counter(i, :) == prd(i)) ~= 1
                    predict_list(i, 1) = nan;
                    predict_list(i, 2) = 1;
                end
            end 
        end
        
    end
end


        
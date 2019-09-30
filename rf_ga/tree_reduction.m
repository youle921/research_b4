addpath('..\dataset')
addpath('..\rf_ga_func')

datalist = {'Vehicle' 'Pima' 'heart' 'glass' 'Satimage'};

tree_num = 200;
cv_num = 2;

path = [char(datetime('now', 'Format', 'MM_dd')) '_tree_reduction_200_OOB'];
mkdir(path)

for i = 1 : length(datalist)

    filename = [datalist{i} '.csv']; 
    T = readtable(filename);
    data = T(:, 1:size(T, 2) - 1);
    answer = T(:, size(T, 2));
    class = table2array(unique(answer));
    
    current_path = [path '\' datalist{i}];
    mkdir(current_path)

    for cv_cnt = 1 : cv_num
        rng(cv_cnt)
        cv = cvpartition(answer{:,1}, 'KFold', 10);

        for cv_trial = 1 : 10
            
            train_data = data(~cv.test(cv_trial), :);
            test_data = data(cv.test(cv_trial), :);
            train_ans = answer(~cv.test(cv_trial), :);
            test_ans = answer(cv.test(cv_trial), :);   

            c_try_num = cv_trial + 10 * (cv_cnt - 1);
            seed = c_try_num;
%             
%             valid_cv = cvpartition(train_ans{:, 1}, 'KFold', 9);
%             valid_data = train_data(valid_cv.test(1), :);
%             valid_ans = train_ans(valid_cv.test(1), :);   
%             train_data = train_data(~valid_cv.test(1), :);
%             train_ans = train_ans(~valid_cv.test(1), :);

            mdl = TreeBagger(tree_num, train_data, train_ans, 'Method', 'classification', ...
                  'OOBPrediction', 'on');

            use_index = 1 : tree_num;
            weight = ones(tree_num, 1);
            
            oob_prd = get_oob_prd_array(mdl, train_data);
            test_prd = get_prd_array(mdl, test_data);
            
            oob_curve = zeros(tree_num, 1);
            oob_curve(1) = accuracy(oob_prd, train_ans, weight);
            test_curve = zeros(tree_num, 1);
            test_curve(1) = accuracy(test_prd, test_ans, weight);
            
            for t = 1 : tree_num - 1
                score = zeros(length(use_index), 1);
                
                for out = 1 : length(use_index)
                    w = weight;
                    w(use_index(out)) = 0;
                    score(out) = accuracy(oob_prd, train_ans, w);
                end
                
                [max_score, max_id] = max(score);

                weight(use_index(max_id)) = 0;
                use_index(max_id) = [];

                oob_curve(t + 1) = max_score;
                test_curve(t + 1) = accuracy(test_prd, test_ans, weight);

            end
            
            csvwrite([current_path '\oob_curve' num2str(c_try_num), '.csv'], oob_curve);
            csvwrite([current_path '\test_curve' num2str(c_try_num), '.csv'], test_curve);

        end
        
    end

    disp([datalist{i} ' finished'])

end

function prd_array = get_prd_array(mdl, data)

    prd_array = zeros(size(data, 1), length(mdl.ClassNames), mdl.NumTrees);
    for t = 1 : mdl.NumTrees
        [~, prd_array(:, :, t)] = predict(mdl.Trees{t}, data);
    end
    
end

function acc = accuracy(prd_array, answer, weight)

    [data_num, class_num, ~] = size(prd_array);
    weight = reshape(repelem(weight, data_num, class_num), data_num, class_num, []);
    [~, prd] = max(sum(prd_array .* weight, 3), 0.1, 2);
    
    acc = sum(prd == answer{:, :}) / height(answer);

end

        
function prd_array = get_oob_prd_array(mdl, data)

    prd_array = get_prd_array(mdl, data);
    for t = 1 : mdl.NumTrees
        oob_array = repmat(mdl.OOBIndices(:, t), 1, length(mdl.ClassNames));
        prd_array(:, :, t) = prd_array(:, :, t) .* oob_array;
    end

end

addpath('..\dataset')
addpath('..\rf_ga_func')

datalist = {'Vehicle' 'Pima' 'heart' 'glass' 'Satimage'};

tree_num = 50;
cv_num = 2;

path = [char(datetime('now', 'Format', 'MM_dd')) '_tree_reduction_50'];
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

        parfor cv_trial = 1 : 10
            
            train_data = data(~cv.test(cv_trial), :);
            test_data = data(cv.test(cv_trial), :);
            train_ans = answer(~cv.test(cv_trial), :);
            test_ans = answer(cv.test(cv_trial), :);   

            c_try_num = cv_trial + 10 * (cv_cnt - 1);
            seed = c_try_num;
            
            valid_cv = cvpartition(train_ans{:, 1}, 'KFold', 9);
            valid_data = train_data(valid_cv.test(1), :);
            valid_ans = train_ans(valid_cv.test(1), :);   
            train_data = train_data(~valid_cv.test(1), :);
            train_ans = train_ans(~valid_cv.test(1), :);

            mdl = TreeBagger(tree_num, train_data, train_ans, 'Method', 'classification', ...
                  'OOBPrediction', 'on');

            use_index = 1 : tree_num;
            weight = ones(tree_num, 1);
            
            valid_prd = get_prd_array(mdl, valid_data);
            test_prd = get_prd_array(mdl, test_data);
            
            valid_curve = zeros(tree_num, 1);
            valid_curve(1) = accuracy(valid_prd, valid_ans, weight);
            test_curve = zeros(tree_num, 1);
            test_curve(1) = accuracy(test_prd, test_ans, weight);
            
            for t = 1 : tree_num - 1
                score = zeros(length(use_index), 1);
                
                for out = 1 : length(use_index)
                    w = weight;
                    w(use_index(out)) = 0;
                    score(out) = accuracy(valid_prd, valid_ans, w);
                end
                
                [max_score, max_id] = max(score);

                weight(use_index(max_id)) = 0;
                use_index(max_id) = [];

                valid_curve(t + 1) = max_score;
                test_curve(t + 1) = accuracy(test_prd, test_ans, weight);

            end
            
            csvwrite([current_path '\valid_curve_' num2str(c_try_num), '.csv'], valid_curve);
            csvwrite([current_path '\test_curve_' num2str(c_try_num), '.csv'], test_curve);

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
    [~, prd] = max(sum(prd_array .* weight, 3), [], 2);
    
    acc = sum(prd == answer{:, :}) / height(answer);

end

% https://jp.mathworks.com/matlabcentral/fileexchange/69812-extreme-learning-machine-for-classification-and-regression
classdef ELM
    properties
        
        activation_function
        
        input_weight
        hidden_weight
        
    end
    
    methods
       
        function obj = ELM(input_size, hidden_size, func)
            
            obj.input_weight = random('Normal', 0, 1, [input_size, hidden_size]);
            
            if strcmp(func, 'sigmoid')
                obj.activation_function = @sigmoid;
            end
            
            if strcmp(func, 'ReLU')
                obj.activation_function = @ReLU;
            end
            
            if strcmp(func, 'RBF')
                obj.activation_function = @RBF;
            end
            
        end
        
        function obj = learning(obj, train_data, train_ans)
            
            H=obj.activation_function(obj.input_weight*train_data');
            obj.hidden_weight=pinv(H') * train_ans;
            
        end
        
        function prd = predict(obj, data)
           
            tmp = obj.activation_function(obj.input_weight * data');
            prd = (tmp * obj.hidden_weight)';

        end
        
        function acc = calc_accuracy(data, answer)
            
            prd = predict(data);
            
            if size(answer, 2) == 1
                acc = mean((prd - answer)^2);
            else
                acc = mean(max(prd, [], 2) == answer);
            end
            
        end
            
    end
    
end

function o = sigmoid(data)

    o = 1 / (1 + exp(-1 * data));

end

function o = ReLU(data)

    o = max(data, 0);
    
end
    
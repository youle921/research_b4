filepath = 'result_t500\';

datalist = {'glass' 'heart' 'Pima' 'Vehicle' 'Satimage'};
method_list = {'oob', 'validation'};

for m = 1 : length(method_list)
    
    savedata = zeros(length(datalist), 3);
    
    for d = 1:length(datalist)
        
        path = [filepath method_list{m} '_' datalist{d} '.csv'];
        savedata(d, :) = mean(csvread(path));
    end
    
    csvwrite([filepath 'mean_' method_list{m} '.csv'], savedata);
    
end

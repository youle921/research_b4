filepath = 'result_OOB_choose\';
ratio = {'0.1' '0.2' '0.3' '0.4' '0.5' '0.6'};

datalist = {'glass' 'heart' 'Pima' 'Vehicle' 'Satimage'};

for d = datalist
    
    savedata = zeros(length(ratio), 3);
    
    for r = 1 : length(ratio)
        savedata(r, 1) = str2double(ratio{r});
        path = [filepath ratio{r} '\oob_' d{:} '.csv'];
        tmp = csvread(path);
        savedata(r, 2:3) = mean(tmp(:, 1:2));
    end
    
    csvwrite([filepath 'mean_' d{:} '.csv'], savedata);
    
end

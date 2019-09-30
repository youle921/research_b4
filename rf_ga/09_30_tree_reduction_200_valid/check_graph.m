datalist = {'Vehicle' 'Pima' 'heart' 'glass' 'Satimage'};

valid_name = '\valid_curve';
test_name = '\test_curve';
ext = '.csv';

for d = datalist
    
    for i = 1 : 20
    
        valid_path = [d{:, :} valid_name num2str(i) ext];
        test_path = [d{:, :} test_name num2str(i) ext];
        
        valid_curve = csvread(valid_path);
        test_curve = csvread(test_path);
        
        close all
        hold on
        
        [~, id] = max(valid_curve);
        plot(valid_curve, '-o', 'MarkerIndices', id, 'MarkerSize', 10 ,'MarkerFaceColor', [0 0.4470 0.7410]);
        
        [~, id] = max(test_curve);
        plot(test_curve, '-o', 'MarkerIndices', id, 'MarkerSize', 10 ,'MarkerFaceColor', [0.8500 0.3250 0.0980]);
        
        legend('valid', 'test', 'Location', 'southwest')

        hold off
        
        pause()
    end
    
    disp(d{:, :});
end
    
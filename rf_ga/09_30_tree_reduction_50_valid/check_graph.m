datalist = {'Vehicle' 'Pima' 'heart' 'glass' 'Satimage'};

valid_name = '\valid_curve_';
test_name = '\test_curve_';
ext = '.csv';

for d = datalist

	valid_average_curve = zeros(50, 1);
	test_average_curve = zeros(50, 1);
    
    for i = 1 : 20
    
        valid_path = [d{:, :} valid_name num2str(i) ext];
        test_path = [d{:, :} test_name num2str(i) ext];
        
        valid_curve = csvread(valid_path);
        test_curve = csvread(test_path);
        
        valid_average_curve = valid_average_curve + valid_curve - valid_curve(1);
        test_average_curve = test_average_curve + test_curve - test_curve(1);
        
        if i == 1
        
%         close all
%         hold on
%         
%
%         plot(valid_curve * 100, 'LineWidth', 2);
%         
%         plot(test_curve * 100, 'LineWidth', 2);
%         
%         legend('validation', 'test', 'Location', 'southwest')
% 
%         hold off
		    set(gca, 'FontSize', 22);
		    xlim([0 50]);
		    xticklabels(string(50 : -10 : 0));
		    
		    xlabel('Number of trees', 'FontSize', 24);
		    ylabel('Accuracy[Åì]', 'FontSize', 24)
		    saveas(gcf, [d{:, :} '_first'], 'meta')
%         
%      end
    end
    
    close all
    hold on
    
    plot(valid_average_curve / 20, 'LineWidth', 2)
    plot(test_average_curve / 20, 'LineWidth', 2)
    
    legend('validation', 'test', 'Location', 'southwest')

    hold off
    set(gca, 'FontSize', 22);
    xlim([0 50]);
    xticklabels(string(50 : -10 : 0));
    if strcmp(d{:,:}, 'Satimage')
        ylim([-0.4 0.1]);
    end
    
    xlabel('Number of trees', 'FontSize', 22);
    ylabel('Difference from Original RF', 'FontSize', 22)
    saveas(gcf, [d{:, :} '_mean'], 'meta')
    
    disp(d{:, :});
end
    
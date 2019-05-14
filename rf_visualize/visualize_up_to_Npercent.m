function visualize_up_to_Npercent(data, score, ratio, figno)

    if nargin < 4
        figno = 1;
    end
    
    [~, score_id] = sort(score, 'descend');
    sort_data = data(score_id, :);
    
    len = size(data, 1);
    num = round(len * ratio);
        
    figure(figno)
    hold on
    
    scatter(sort_data(num + 1:len, 1), sort_data(num + 1:len, 2), 200, 'MarkerEdgeColor', [0.5 0.5 0.5], 'MarkerFaceColor', [0.8 0.8 0.8])
    scatter(sort_data(1:num, 1), sort_data(1:num, 2), 200, 'r', 'filled')
    
    axis square
    grid on
    
    set(gca, 'Fontsize', 24);
    
end


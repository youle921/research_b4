function visualize_bottom_to_up(data, score)

    colortable = get_colortable(score);
    
    [~, score_id] = sort(score);
    
    hold on
    
    for i = 1:length(score_id)
        id = score_id(i);
        scatter(data(id, 1), data(id, 2), 200, 'MarkerEdgeColor', max(colortable(id, :) - 0.55, 0), 'MarkerFaceColor', colortable(id, :))
    end
    
    axis square
    grid on
    
    set(gca, 'Fontsize', 24);
    
end


function draw_default_scatter(fig_no, data, color_table)

    figure(fig_no);
    scatter(data(:, 1), data(:, 2), 200, color_table, 'filled')

end


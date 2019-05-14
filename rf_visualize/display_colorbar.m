function display_colorbar(score, label)

    table = (0:1000) *0.001;
    table = [table; table; table].';
    colorbar_array = get_colortable(table);

%     現在のグラフに指定した色でカラーバーを表示
    colormap(colorbar_array)
%     現在のグラフのカラーバーの範囲を指定
    caxis([min(score) max(score)])
    
    c = colorbar;
    c.Label.String = label;

end


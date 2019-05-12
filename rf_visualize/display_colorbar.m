function display_colorbar(score)

    table = (0:1000) *0.001;
    table = [table; table; table].';
    colorbar_array = get_colortable(table);

%     現在のグラフに指定した色でカラーバーを表示
    colormap(colorbar_array)
%     現在のグラフのカラーバーの範囲を指定
    caxis([min(score)*100 max(score)*100])
    
    c = colorbar;
    c.Label.String = "Accuracy[％]";

end


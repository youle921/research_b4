function display_colorbar(score)

    table = (0:1000) *0.001;
    table = [table; table; table].';
    colorbar_array = get_colortable(table);

%     ���݂̃O���t�Ɏw�肵���F�ŃJ���[�o�[��\��
    colormap(colorbar_array)
%     ���݂̃O���t�̃J���[�o�[�͈̔͂��w��
    caxis([min(score)*100 max(score)*100])
    
    c = colorbar;
    c.Label.String = "Accuracy[��]";

end


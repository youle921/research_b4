% https://jp.mathworks.com/matlabcentral/answers/215624-how-do-i-get-the-depth-of-a-tree

function depth_list = get_rf_trees_depth(trees)

    len = length(trees);
    depth_list = zeros(len, 1);
    for i = 1 : len
        parent = trees{i}.Parent;
        node = parent(end);
        while node~=0
            depth_list(i) = depth_list(i) + 1;
            node = parent(node);
        end
    end
    
end




function params = set_ga_params(varargin)

%% default setting
    params.tree_num = 50;
    params.p_num = 50;
    params.c_num = 50;
    params.gen_num = 1000;
    
%% check custom setting
    if nargin ~= 0
        
        for i = 1 : nargin/2

            if strcmp(varargin{i * 2 - 1}, 'tree_num')
                params.tree_num = varargin{i * 2};
            elseif strcmp(varargin{i * 2 - 1}, 'p_num')
                params.p_num = varargin{i * 2};
            elseif strcmp(varargin{i * 2 - 1}, 'c_num')
                params.c_num = varargin{i * 2};
            else
                params.gen_num = varargin{i * 2};
            end
            
        end
        
    end
    
end
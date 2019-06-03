function choose_id = SOM_choose(data, ratio)

c_num = round(ratio * size(data, 1));

% Parameter for SOM -------------------------------------------------------
SOMnet.width  = round(c_num^0.5);  % width of map
SOMnet.height = round(c_num^0.5);  % height of map
SOMnet.sig = 0.5;    % Variance
SOMnet.nNode = SOMnet.height * SOMnet.width; % Number of nodes in network
SOMnet.weight = rand(SOMnet.nNode, 2);       % Initialization of weight

for k=1:size(data,2)
    mmin = min(data(:,k));
    mmax = max(data(:,k));
    data(:,k) = (data(:,k)-mmin) ./ (mmax-mmin);
end

MIter = 30;     % Maximum number of iterations
for nitr = 1:MIter
    SOMnet = SOM(data, SOMnet, nitr, MIter);
end

choose_id = knnsearch(data, SOMnet.weight);
    
end


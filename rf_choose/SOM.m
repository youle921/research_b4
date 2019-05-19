function net = SOM(Data, net, nitr, MIter)

width = net.width;
height = net.height;
weight = net.weight;
sig = net.sig;


% Scale learning paramters by elapsed time                        
tfrac = nitr / MIter;
SIG = sig*tfrac;
alpha_t = 1-nitr/MIter;


for sampleNum = 1:size(Data,1)
    
    % Select Input
    pattern = Data(sampleNum,:);
    
    % Find best matching unit between nodes of SOM and l-th input data.
    dist = pdist2(pattern, weight);
    [~, bmuindex] = min(dist);          % Index of best matching unit
    neighbor_nodes = getNeighborNode(bmuindex, height, width, 1); % Find neighbor nodes of best matching unit.
    
    % udpate the weights
    for k = 1:numel(neighbor_nodes)
        
        sIndex = neighbor_nodes(k);  % index of target neighbor node
        
        h = exp(-sum((weight(bmuindex,:)-weight(sIndex,:)).^2) / (2*SIG^2));
        dW = alpha_t * h * (pattern - weight(sIndex,:));
        
        weight(sIndex,:) = weight(sIndex,:) + dW; 
    end
    
    
    
end


net.weight = weight;

end
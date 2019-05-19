function net = GNG(Data, net)

    
    % Parameters
    epsilon_b = net.epsilon_b;
    epsilon_n = net.epsilon_n;
    maxAge = net.maxAge;         % max age
    lambda = net.lambda;         % lambda  % cycle for inserting a new node
    alpha = net.alpha;           % q and f units error reduction constant.
    delta = net.delta;           % Error reduction factor.
    maxNode = net.maxNode;
    
    weight = net.weight;
    Err  = net.Err;   % Error Vector.
    edge = net.edge;  % connections (edges) matrix.
    age  = net.age;   % age matrix. 
    
    
    
    % Following "STEP" is corresponding to algorithm in original GNG paper.
    
    % Initialization
    if size(weight,1) == 0
        Ni = 2; % Step 0: Start with two neural units (nodes) selected from input data:
        Xmin = min(Data);
        Xmax = max(Data);
        for i = 1:Ni
            weight(i,:) = unifrnd(Xmin, Xmax);
        end
        
        Err = [0; 0];
    end
    
    
    
    for sampleNum = 1:size(Data,1)
        
        
        % Step 1: Select Input
        pattern = Data(sampleNum,:);
        
        % Step 2: Find the two nearest units s1 and s2 to the new data sample.
        d = pdist2(weight,pattern);
        d = d';
        [~, SortOrder] = sort(d);
        s1 = SortOrder(1);
        s2 = SortOrder(2);

        % Steps 3: Increment the age of all edges emanating from s1.
        age(s1, :) = age(s1, :) + 1;
        age(:, s1) = age(:, s1) + 1;
        
        
        % Step 4: Add the squared distance to a local error counter variable:
        Err(s1) = Err(s1) + d(s1)^2;    % ||w-x||^2

        % Step 5: Move s1 and its topological neighbors towards x.
        weight(s1,:) = weight(s1,:) + epsilon_b * (pattern - weight(s1,:));
        N_s1 = find(edge(s1,:)==1);
        for j=N_s1
            weight(j,:) = weight(j,:) + epsilon_n * (pattern - weight(j,:)); % for Neighbor nodes which are connecting to s1.
        end
        

        % Step 6:
        % If s1 and s2 are connected by an edge, set the age of this edge to zero.
        % If such an edge does not exist, create it.
        edge(s1,s2) = 1;
        edge(s2,s1) = 1;
        age(s1,s2) = 0;
        age(s2,s1) = 0;

        % Step 7(1):
        % Remove edges from node if age>maxAge.
        edge(age>maxAge) = 0;
        nNeighbor = sum(edge);

        % Step 7(2):
        % Remove isolated node if the node has no emanating edges.
        AloneNodes = (nNeighbor==0);
        edge(AloneNodes, :) = [];
        edge(:, AloneNodes) = [];
        age(AloneNodes, :) = [];
        age(:, AloneNodes) = [];
        weight(AloneNodes, :) = [];
        Err(AloneNodes) = [];
        

        % Step 8: Node Insertion Procedure.
        %   w_r: new node
        %   w_q: node which has maximum accumulated error
        %   w_f: neighbor nodes of q
        if mod(sampleNum, lambda) == 0 && maxNode >= size(weight,1)
            [~, q] = max(Err);
            [~, f] = max(edge(:,q).*Err);
            r = size(weight,1) + 1;
            weight(r,:) = (weight(q,:) + weight(f,:))/2;   % Insert node r between nodes q and f
            edge(q,f) = 0;  % delete edges between node q and f (q--f -> q  f)
            edge(f,q) = 0;
            edge(q,r) = 1;  % create edges between node q and r (q-r f)
            edge(r,q) = 1;
            edge(r,f) = 1;  % create edges between node f and r (q-r-f)
            edge(f,r) = 1;
            age(r,:) = 0;   % initialize age for node r
            age(:,r) = 0;
            Err(q) = alpha * Err(q);  % update error by constant alpha
            Err(f) = alpha * Err(f);
            Err(r) = Err(q);
            
        end

        % Step 9: Decrease the error of all units.
        Err = delta*Err;

    end


    net.weight = weight;
    net.Err = Err;
    net.edge = edge;
    net.age = age;
    

end


function choose_id = GNG_choose(data, ratio)

c_num = round(ratio * size(data, 1));

% Parameters for GNG ------------------------------------------------------
GNGnet.maxNode = c_num - 1;     % Maximum number of clusters
GNGnet.weight = [];       % Center of cluster ÉmÅ[ÉhÇÃç¿ïW
GNGnet.Err = [];          % Error Vector
GNGnet.edge = zeros(2,2); % Edge between clusters
GNGnet.age = zeros(2,2);  % Age of edge

GNGnet.epsilon_b = 0.2;   % Learning coefficient
GNGnet.epsilon_n = 0.05;  % Learning coefficient
GNGnet.maxAge = 50;       % Maximum cluster age
GNGnet.lambda = 100;      % Cycle for topology reconstruction (Denoising)
GNGnet.alpha = 0.5;       % Nodes q and f units error reduction constant
GNGnet.delta = 0.9;       % Error reduction coefficient

for k=1:size(data,2)
    mmin = min(data(:,k));
    mmax = max(data(:,k));
    data(:,k) = (data(:,k)-mmin) ./ (mmax-mmin);
end

MIter = 30;     % Maximum number of iterations
for nitr = 1:MIter
    GNGnet = GNG(data, GNGnet);
end

choose_id = knnsearch(data, GNGnet.weight);
    
end


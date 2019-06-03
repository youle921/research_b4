
% 
% Demo for Topological Models
% Please run this file
% 

whitebg('black')
% whitebg('white')

% 
% 1. Growing Neural Gas (GNG)
% B. Fritzke,
% "A growing neural gas network learns topologies",
% Advances in neural information processing systems, Vol.7, pp.625-632, 1995.
%

% 
% 3. Self-Organizing Map (SOM)
% T. Kohonen,
% "Self-organized formation of topologically correct feature maps",
% Biological cybernetics, Vol.43, No.1, pp.59-69, 1982.
% 



% Parameters for GNG ------------------------------------------------------
GNGnet.maxNode = 500;     % Maximum number of clusters
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


% Parameter for SOM -------------------------------------------------------
SOMnet.width  = 10;  % width of map
SOMnet.height = 10;  % height of map
SOMnet.sig = 0.5;    % Variance
SOMnet.nNode = SOMnet.height * SOMnet.width; % Number of nodes in network
SOMnet.weight = rand(SOMnet.nNode, 2);       % Initialization of weight



% 2D Data setting
nData = 200; % Number of data
NR = 0.0;     % Noise Rate [0-1]

% 2D Dataset [numSamples, numFeatures]
% data = corners(nData);
% data = crescentfullmoon(nData);
%data = halfkernel(nData);
% data = outlier(nData);
% data = twospirals(nData);
 %data = rings(nData);

data = prd_Y;
data = [data(:,1) data(:,2)];


% Data Normalization [0-1]
for k=1:size(data,2)
    mmin = min(data(:,k));
    mmax = max(data(:,k));
    data(:,k) = (data(:,k)-mmin) ./ (mmax-mmin);
end

% Add Noise
DATA = [data; rand(nData*NR,2)];


% Randamize data
ran = randperm(size(DATA,1));
DATA = DATA(ran,:);
DATA = DATA(1:nData,:);




MIter = 50;     % Maximum number of iterations

for nitr = 1:MIter
    fprintf('Iterations: %d/%d\n',nitr,MIter);
    
    % GNG
    GNGnet = GNG(DATA, GNGnet);
    
    % SOM
    SOMnet = SOM(DATA, SOMnet, nitr, MIter);
    
    
    figure(1); myPlotGNG(DATA, GNGnet, 'GNG');
    figure(2); myPlotSOM(DATA, SOMnet, 'SOM');

end


figure(1); myPlotGNG(DATA, GNGnet, 'GNG');
figure(2); myPlotSOM(DATA, SOMnet, 'SOM');



% Reinforcement‑Based Clustering (practical MATLAB version)
% Inputs:
%   X         : M x N data matrix
%   K         : number of clusters
%   eps       : epsilon-greedy policy

% Outputs:
%   P         : M x K assignment probabilities, rows sum to 1
%   Y         : K x N cluster centers

% ------------------ initialization -------------------
close all; clc; 
idx = 4; [X,K,T_P,M,N] = data_RLClustering_ModelFree(idx); 
[X, ~, ~] = zscore(X); 
alp = 1; count = ones(length(X),K); eps = 0.1;
[X, mu, sigma] = zscore(X); Q = zeros(length(X),K); 
Qbar = zeros(length(X),K); indx = 1;

Tmin = 0.001; alpha = 0.99; PERTURB = 0.0001; STOP = 1e-2;
T = 5; Px = (1/M)*ones(M,1); Y = load('Ydata_idx4.mat'); %Y = repmat(Px'*X, [K,1]);
MaxOuter = 5000; P = 1/K*ones(M,K);

Memory.i = []; Memory.j = []; Memory.k = [];

% ------------------ annealing loop over beta ----------

while T > Tmin
    
    % ===== "policy" loop: update Pi given current Y =====
    for t = 1 : MaxOuter
        
        values = 1:length(X);
        if indx <= length(X)
            i = indx;
        else
            indx = 1;
            i = indx;
        end
        indx = indx + 1;
        
        % epsilon-greedy policy 
        if rand < eps
            j = randi([1, K]);
        else
            values = 1 : K; j = randsample(values, 1, true, P(i,:));
        end
        [k, dist] = environment_dataClustering(i,j,T_P,K,X,Y);
        count(i,j) = count(i,j) + 1;
        alpha = 1/count(i,j);
        Q(i,j) = Q(i,j) + alpha*(dist - Q(i,j));

        Qbar(i,:) = Q(i,:) - min(Q(i,:));
        P(i,:) = exp(-(1/T)*Qbar(i,:));
        P(i,:) = P(i,:)./(repmat(sum(P(i,:)),[1 K]));
        
        if length(Memory.i) < 10000
            Memory.i = [Memory.i i]; 
            Memory.j = [Memory.j j]; 
            Memory.k = [Memory.k k];
        else
            Memory.i(1) = []; Memory.j(1) = []; Memory.k(1) = [];
            Memory.i = [Memory.i i]; 
            Memory.j = [Memory.j j]; 
            Memory.k = [Memory.k k];
        end

    end
    disp(T);
    T = T*0.98;
end
close all;
scatter(X(:,1),X(:,2),90,'filled','MarkerEdgeColor',[0 0.5 0.5],...
'MarkerFaceColor',[0 0.7 0.7],'LineWidth',1.5);
xlim([-2 2]); ylim([-2 2]); axis square; box on;
set(gca, 'FontSize', 25); set(gca, 'LineWidth', 1.0);
xticks([-5 0 5]); yticks([-5 0 5]);
hold on; scatter(Y.Y(:,1),Y.Y(:,2),500,'p','MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');

P_idx = round(P);
idx = cell(K,1);
for k = 1 : K
    idx{k} = find(P(:,k)==1);
    col = rand(1,3);
    scatter(X(idx{k},1),X(idx{k},2),90,'filled','MarkerEdgeColor','k',...
        'MarkerFaceColor',col,'LineWidth',0.25);
    scatter(Y.Y(k,1),Y.Y(k,2),500,'p','MarkerEdgeColor', 'k', 'MarkerFaceColor', col, 'LineWidth',2);
end



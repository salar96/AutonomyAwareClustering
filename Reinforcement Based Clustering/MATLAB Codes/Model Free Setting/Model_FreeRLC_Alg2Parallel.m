% Reinforcement‑Based Clustering
% Inputs:
%   X         : M x N data matrix
%   K         : number of clusters
%   eps       : epsilon-greedy policy

% Outputs:
%   P         : M x K assignment probabilities, rows sum to 1
%   Y         : K x N cluster centers

% ------------------ initialization -------------------
tic; close all; clc; 
idx = 4; [X,K,T_P,M,N] = data_RLClustering_ModelFree(idx); 
[X, ~, ~] = zscore(X); 
alp = 1; count = ones(length(X),K); eps = 0.1;
[X, mu, sigma] = zscore(X); Q = zeros(length(X),K); 
Qbar = zeros(length(X),K); indx = 1;

Tmin = 0.001; alpha = 0.99; PERTURB = 0.0001; STOP = 1e-2;
T = 5; Px = (1/M)*ones(M,1); Y = repmat(Px'*X, [K,1]);
MaxOuterP = 5000; MaxOuterY = 5000; P = 1/K*ones(M,K);
gamma = 0.1;

Memory.i = []; Memory.j = []; Memory.k = [];

% ------------------ annealing loop over beta ----------

while T > Tmin
    
    % ===== "policy" loop: update P given current Y =====
    for t = 1 : MaxOuterP
        
        Q_old = Q;
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
        
        if length(Memory.i) < 100000
            Memory.i = [Memory.i i]; 
            Memory.j = [Memory.j j]; 
            Memory.k = [Memory.k k];
        else
            Memory.i(1) = []; Memory.j(1) = []; Memory.k(1) = [];
            Memory.i = [Memory.i i]; 
            Memory.j = [Memory.j j]; 
            Memory.k = [Memory.k k];
        end

        if norm(Q_old - Q) < 1e-04
           break;
        end
    end

    % ===== "centroid" loop: update Y given current P and stochasticity =====
    gamma = 0.001;
    Pij = cell(K,1); Par_memory = cell(K,1); X_par = cell(K,1);
    for k = 1 : K
        Pij{k} = P; Par_memory{k} = Memory; X_par{k} = X;
    end
    count1 = cell(K,1); count1{1} = 1; count1{2} = 1; count1{3} = 1; count1{4} = 1;
    parfor l = 1 : K
        for t = 1 : MaxOuterY
            idx = randi([1 length(Par_memory{l}.i)]);
            i = Par_memory{l}.i(idx); j = Par_memory{l}.j(idx); k = Par_memory{l}.k(idx);
            if k == l
                Y(l,:) = Y(l,:) - gamma*Pij{l}(i,j)*(Y(l,:) - X_par{l}(i,:));
                count1{l} = count1{l} + 1;
            end
        end
    end
    ttl =  count1{1} + count1{2} + count1{3} + count1{4};
    fprintf('%d %d %d %d %d', count1{1}, count1{2}, count1{3}, count1{4});
    fprintf(' = '); fprintf('%d', ttl);
    fprintf('\n');
    disp(T);
    T = T*0.99;
end
close all;

P_idx = round(P);
idx = cell(K,1);
for k = 1 : K
    idx{k} = find(P(:,k)==1);
    col = rand(1,3);
    scatter(X(idx{k},1),X(idx{k},2),90,'filled','MarkerEdgeColor','k',...
        'MarkerFaceColor',col,'LineWidth',0.25); hold on;
    scatter(Y(k,1),Y(k,2),500,'p','MarkerEdgeColor', 'k', 'MarkerFaceColor', col, 'LineWidth',2);
end
xlim([-2 2]); ylim([-2 2]); axis square; box on;
set(gca, 'FontSize', 25); set(gca, 'LineWidth', 1.0);
xticks([-2 0 2]); yticks([-2 0 2]); hold off;
save('results_Alg2.mat');
toc;
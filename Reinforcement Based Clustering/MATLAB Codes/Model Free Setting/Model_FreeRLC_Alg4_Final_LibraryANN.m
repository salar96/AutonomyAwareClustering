%% Algorithm 4: Deep Clustering Algorithm (large N/ continuous X) - In-built ANN library
clear; clc; rng(1);
MaxInnerTheta = 200;                % policy update loop max iterations
MaxInnerY = 200;                    % max iterations in Y update loop
MaxInnerThetaArr = [100, 150, 200, 250];
MaxInnerYArr = [100, 150, 200, 250];
% ------------------ initialization -------------------
for inner1 = 1 : length(MaxInnerThetaArr)
    for inner2 = 1 : length(MaxInnerYArr)
        close all; clc; 
        idx = 4; [X,K,T_P,M,N] = data_RLClustering_ModelFree(idx);
        [X,mu,sig] = zscore(X);
        
        T = 10; Tmin = 0.01; tau = 0.96;    % annealing parameters
        eps = 0.1;                          % for epsilon greedy policy 
        Sz_miniBatch = 512;                 % size of the minibatch for both nets and Y
        buf_cap = 200000;                   % memory or replay capacity
        
        
        H_target = MaxInnerTheta/10;        % steps between target net updates
        
        alpha = 0.005;                      % learning rate for Y SGD
        gamma = 0.001;                      % learning rate for policy SGD update
        
        Px = (1/M)*ones(M,1);               % weight for each user data point
        [idx_clust, Y] = kmeans(X,K);        % Starting with Y
        
        P = 1/K*ones(M,K);                  % Policy 
        
        
        idx_input = randi([1 length(X)],[Sz_miniBatch,1]);
        init_trainInput = zeros(Sz_miniBatch,2*N);
        init_trainOutput = zeros(Sz_miniBatch,1);
        for i = 1 : Sz_miniBatch
            init_trainInput(i,:) = [X(idx_input(i),:) Y(idx_clust(idx_input(i)),:)];
            init_trainOutput(i) = norm(X(idx_input(i),:)-Y(idx_clust(idx_input(i)),:))^2;
        end
        init_trainInput = init_trainInput';
        init_trainOutput = init_trainOutput';
        Y = repmat(Px'*X, [K,1]);
        
        net = feedforwardnet([32 16], 'trainlm');
        net.trainParam.epochs = 250;
        net.trainParam.showWindow = false;
        net = train(net, init_trainInput, init_trainOutput);
        
        net_target = net;
        
        memory.i = zeros(buf_cap,1); memory.j = zeros(buf_cap,1);
        memory.k = zeros(buf_cap,1); memory.d = zeros(buf_cap,1);
        
        buf_n = 0;                          % current size of the memory
        circ_ptr = 1;                       % circular pointer
        thetaPrev = getwb(net);
        K_consec = 20;
        epsTheta = 1e-06;
        epsY = 1e-04;


        MaxInnerTheta = MaxInnerThetaArr(inner1);
        MaxInnerY = MaxInnerYArr(inner2);
    
        while T >= Tmin
            
            % ===== "policy" loop: update P given current Y =====
            smallCnt = 0;
            for t = 1 : MaxInnerTheta
                i = randi(M); % sampling a user data point
        
                d_bar = zeros(K,1);
                for j = 1 : K
                    %d_bar(j) = nn_forward(theta_target, X(i,:), Y(j,:));
                    d_bar(j) = net_target([X(i,:),Y(j,:)]');
                end
                num = exp(-(1/T)*d_bar);
                den = sum(num);
                P(i,:) = num/den;
        
                % epsilon-greedy method for action selection
                if rand < eps
                    j = randi(K);
                else
                    [~,j] = max(P(i,:));
                end
                
                k = randsample(1:K, 1, true, T_P(:,j,i));   % sample the cluster k using T_P
                dist = norm(X(i,:) - Y(k,:))^2;            
                if buf_n < buf_cap
                    loc = circ_ptr; buf_n = buf_n + 1;
                else
                    loc = circ_ptr;
                end
                circ_ptr = circ_ptr + 1; 
                if circ_ptr > buf_cap
                    circ_ptr = 1;
                end
        
                memory.i(loc) = i; memory.j(loc) = j; 
                memory.k(loc) = k; memory.d(loc) = dist;
        
                % train theta when enough data is available
                if buf_n >= Sz_miniBatch
                    miniBatch = randi(buf_n, Sz_miniBatch, 1);   % uniformly sample
                    miniBatchDataX = zeros(Sz_miniBatch,2*N);
                    miniBatchDataY = zeros(Sz_miniBatch,1);
                    for b = 1 : Sz_miniBatch
                        ii = memory.i(miniBatch(b));
                        jj = memory.j(miniBatch(b));
                        d_t = memory.d(miniBatch(b));
                        miniBatchDataX(b,:) = [X(ii,:), Y(jj,:)];
                        miniBatchDataY(b) = d_t;
                    end
                    net.trainParam.epochs = 2; net.divideFcn = 'dividetrain';
                    net = train(net, miniBatchDataX', miniBatchDataY');
                end
                thetaNow  = getwb(net);
                dthetaRel = norm(thetaNow - thetaPrev) / (norm(thetaPrev) + 1e-12);
                if dthetaRel < epsTheta
                    smallCnt = smallCnt + 1;
                else
                    smallCnt = 0;
                end
                if smallCnt >= K_consec
                    net_target = net;
                    break;
                end
                thetaPrev = thetaNow;
        
                % target network sunc
                if mod(t, H_target) == 0
                    net_target = net;
                end
                %disp(t);
            end
        
            % ===== "centroid" loop: update Y given current P and stochasticity =====
        
            Y_old = Y;
            for t = 1 : MaxInnerY
                idx = randi([1 buf_n]);
                ii = memory.i(idx); jj = memory.j(idx); kk = memory.k(idx);
                d_hat = zeros(1,K);
                for j1 = 1 : K
                    d_hat(j1) = net_target([X(ii,:), Y(j1,:)]');
                end
                num = exp(-(1/T)*d_hat);
                den = sum(num);
                Pij = num/den;
                Y(kk,:) = Y(kk,:) - alpha*Pij(jj)*(Y(kk,:) - X(ii,:));
                if norm(Y_old(kk,:) - Y(kk,:)) < epsY
                    count_Y = count_Y + 1;
                else
                    count_Y = 0;
                end
                %disp(count_Y);
                if count_Y >= K_consec
                    break;
                end
                Y_old = Y;
            end
            
            net_target_ij = cell(K,1); Par_memory = cell(K,1); 
            X_par = cell(K,1); Y_par = cell(K,1); net_Y = cell(K,1);
            comp_Pij_Y = cell(K,1);
            for k = 1 : K
                net_target_ij{k} = net_target; X_par{k} = X; Y_par{k} = Y(k,:);
                Par_memory{k} = memory; net_Y{k} = net; 
                comp_Pij_Y{k} = Y;
            end
            
            count_Y = zeros(K,1);
            parfor l = 1 : K
                Y_old = Y_par{l};
                count_Y(l) = 0;
                for t = 1 : MaxInnerY
                    idx = randi([1 buf_n]);
                    i = Par_memory{l}.i(idx); j = Par_memory{l}.j(idx); k = Par_memory{l}.k(idx);
                    d_hat = zeros(K,1);
                    for j1 = 1 : K
                        d_hat(j1) = net_target_ij{l}([X_par{l}(i,:), comp_Pij_Y{l}(j1,:)]');
                    end
                    num = exp(-(1/T)*d_hat); den = sum(num); 
                    Pij = num/den;
                    if k == l
                        Y_par{l} = Y_par{l} - alpha*Pij(j)*(Y_par{l} - X_par{l}(i,:));
                    else
                        continue;
                    end
                    if norm(Y_old - Y_par{l})/(norm(Y_par{l}) + 1e-12) < epsY
                        count_Y(l) = count_Y(l) + 1;
                    else
                        count_Y(l) = 0;
                    end
                    if count_Y(l) >= K_consec
                        break;
                    end
                    Y_old = Y_par{l};
                end
            end
            for j = 1 : K
                Y(j,:) = Y_par{j};
            end
            disp(T);
            T = tau*T;
        end
        
        P_idx = zeros(M,1);
        for i = 1 : M
            d_bar = zeros(K,1);
            for j = 1 : K
                %d_bar(j) = nn_forward(theta_target, X(i,:), Y(j,:));
                d_bar(j) = net_target([X(i,:),Y(j,:)]');
            end
            num = exp(-(1/T)*d_bar);
            den = sum(num);
            P(i,:) = num/den;
            [~,P_idx(i)] = max(d_bar);
        end
        
        idx = cell(K,1);
        for k = 1 : K
            idx{k} = find(P_idx==k);
            col = rand(1,3);
            scatter(X(idx{k},1),X(idx{k},2),90,'filled','MarkerEdgeColor','k',...
                'MarkerFaceColor',col,'LineWidth',0.25); hold on;
            scatter(Y(k,1),Y(k,2),500,'p','MarkerEdgeColor', 'k', 'MarkerFaceColor', col, 'LineWidth',2);
        end
        xlim([-2 2]); ylim([-2 2]); axis square; box on;
        set(gca, 'FontSize', 25); set(gca, 'LineWidth', 1.0);
        xticks([-2 0 2]); yticks([-2 0 2]); hold off;
        title("MaxInnerTheta = " + MaxInnerTheta + " MaxInnerY = " + MaxInnerY);
        name = ['Run_' num2str(MaxInnerTheta) '_' num2str(MaxInnerY) '.png'];
        print(gcf, name, '-dpng', '-r600');
        name = ['Run_Results_' num2str(MaxInnerTheta) '_' num2str(MaxInnerY) '.mat'];
        save(name)
    end
end
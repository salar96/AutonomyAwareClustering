%% Algorithm 4: Deep Clustering Algorithm (large N/ continuous X) - In-built ANN library
tic;clear all; clc; rng(1);
MaxInnerTheta = 200;                % policy update loop max iterations
MaxInnerY = 200;                    % max iterations in Y update loop
MaxInner = [500, 250];
NN_update = 50;
%MaxInnerThetaArr = [150, 200, 250];
%MaxInnerYArr = [150, 200, 250];
v = VideoWriter('CheckingMiniBatchSGDPerformanceRun2.mp4', 'Motion JPEG 2000');  % or 'Motion JPEG AVI'
v.FrameRate = 10;  % frames per second
open(v);
% ------------------ initialization -------------------
for inner = 1 : size(MaxInner,1)
    
    close all; clc; 
    idx = 4; [X,K,T_P,M,N] = data_RLClustering_ModelFree(idx);
    [X,mu1,sig] = zscore(X);
    
    T = 0.5; Tmin = 0.0001; tau1 = 0.9;    % annealing parameters
    eps = 0.99;                          % for epsilon greedy policy 
    Sz_miniBatch = 256;                 % size of the minibatch for both nets and Y
    buf_cap = 500;                   % memory or replay capacity
    
    
    H_target = MaxInnerTheta/10;        % steps between target net updates
    
    alpha = 0.01;                      % learning rate for Y SGD
    mu = 0.9;
    gamma = 0.001;                      % learning rate for policy SGD update
    tau = 1;
    
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
    Y = repmat(Px'*X, [K,1]) + 0.1*randn(K,N);
    
    net = feedforwardnet([16 8], 'trainlm');
    net.performParam.regularization = 0.3;
    %net = feedforwardnet([10 10], 'trainlm');
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

    MaxInnerTheta = MaxInner(inner,1);
    MaxInnerY = MaxInner(inner,2);
    count = 1;
    while T >= Tmin
        
        % ===== "policy" loop: update P given current Y =====
        eps = 0.99;
        smallCnt = 0;
        for t = 1 : MaxInnerTheta * 1
            i = randi(M); % sampling a user data point
    
            d_bar = zeros(K,1);
            for j = 1 : K
                %d_bar(j) = nn_forward(theta_target, X(i,:), Y(j,:));
                d_bar(j) = net_target([X(i,:),Y(j,:)]');
            end
            d_bar2 = d_bar - min(d_bar,[],2);
            num = exp(-(1/T)*d_bar2);
            den = sum(num);
            P(i,:) = num/den;
    
            % epsilon-greedy method for action selection
            if rand < eps
                j = randi(K);
            else
                %[~,j] = max(P(i,:));
                j = randsample(1:K, 1, true, P(i,:));
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
            if buf_n >= Sz_miniBatch && mod(t,25) == 0
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
                net.trainParam.epochs = 100; net.divideFcn = 'dividetrain';
                [net,tr] = train(net, miniBatchDataX', miniBatchDataY');
            end
            thetaNow  = getwb(net);
            dthetaRel = norm(thetaNow - thetaPrev) / (norm(thetaPrev) + 1e-12);
            if dthetaRel < epsTheta
                smallCnt = smallCnt + 1;
            else
                smallCnt = 0;
            end
            %if smallCnt >= K_consec
            %    net_target = net;
            %    break;
            %end
            thetaPrev = thetaNow;
    
            % target network sunc
            if mod(t, 100) == 0
                net_target = net;
            end
            %disp(t);
            eps = eps*0.99;
        end
        % if buf_n >= Sz_miniBatch
        %     t1 = miniBatchDataX(:,2);
        %     t2 = miniBatchDataY;
        %     y =  net(miniBatchDataX'); 
        %     rsq = @(y,t1) 1 - sum((t1(:) - y(:)).^2) / sum((t1(:) - mean(t1(:))).^2);
        %     R2_train = rsq(y, t2);
        %     disp(R2_train);
        % end
        % ===== "centroid" loop: update Y given current P and stochasticity =====
    
        % Y_old = Y;
        % for t = 1 : MaxInnerY
        %     idx = randi([1 buf_n]);
        %     ii = memory.i(idx); jj = memory.j(idx); kk = memory.k(idx);
        %     d_hat = zeros(1,K);
        %     for j1 = 1 : K
        %         d_hat(j1) = net_target([X(ii,:), Y(j1,:)]');
        %     end
        %     num = exp(-(1/T)*d_hat);
        %     den = sum(num);
        %     Pij = num/den;
        %     Y(kk,:) = Y(kk,:) - alpha*Pij(jj)*(Y(kk,:) - X(ii,:));
        %     if norm(Y_old(kk,:) - Y(kk,:)) < epsY
        %         count_Y = count_Y + 1;
        %     else
        %         count_Y = 0;
        %     end
        %     %disp(count_Y);
        %     if count_Y >= K_consec
        %         break;
        %     end
        %     Y_old = Y;
        % end
        
        %Y = Y + 0.1*randn(K,N);
        net_target_ij = cell(K,1); Par_memory = cell(K,1); 
        X_par = cell(K,1); Y_par = cell(K,1); net_Y = cell(K,1);
        comp_Pij_Y = cell(K,1);
        for k = 1 : K
            net_target_ij{k} = net_target; X_par{k} = X; Y_par{k} = Y(k,:);
            Par_memory{k} = memory; net_Y{k} = net; 
            comp_Pij_Y{k} = Y;
        end
        % 
        % count_Y = zeros(K,1);
        % parfor l = 1 : K
        %     Y_old = Y_par{l};
        %     count_Y(l) = 0;
        %     V_t = zeros(size(Y_old));
        %     for t = 1 : MaxInnerY
        %         idx = randi([1 buf_n]);
        %         i = Par_memory{l}.i(idx); j = Par_memory{l}.j(idx); k = Par_memory{l}.k(idx);
        %         d_hat = zeros(K,1);
        %         for j1 = 1 : K
        %             d_hat(j1) = net_target_ij{l}([X_par{l}(i,:), comp_Pij_Y{l}(j1,:)]');
        %         end
        %         num = exp(-(1/T)*d_hat); den = sum(num); 
        %         Pij = num/den;
        %         if k == l
        %             %V_t = mu*V_t - alpha*Pij(j)*(Y_par{l} - X_par{l}(i,:));
        %             %Y_par{l} = Y_par{l} + V_t;
        %             Y_par{l} = Y_par{l} - alpha*Pij(j)*(Y_par{l} - X_par{l}(i,:));
        %         else
        %             continue;
        %         end
        %         if norm(Y_old - Y_par{l})/(norm(Y_par{l}) + 1e-12) < epsY
        %             count_Y(l) = count_Y(l) + 1;
        %         else
        %             count_Y(l) = 0;
        %         end
        %         if count_Y(l) >= K_consec
        %             break;
        %         end
        %         Y_old = Y_par{l};
        %     end
        % end

        count_Y = zeros(K,1);
        for l = 1 : K
            Y_old = Y_par{l};
            count_Y(l) = 0;
            V_t = zeros(size(Y_old));
            for t = 1 : MaxInnerY
                %idx = randi([1 buf_n],100,1);
                if buf_n < 100
                    idx = randperm(buf_n,buf_n);
                else
                    idx = randperm(buf_n,100);
                end
                i_idx = Par_memory{l}.i(idx); j_idx = Par_memory{l}.j(idx); k_idx = Par_memory{l}.k(idx);
                idx_l = find(k_idx == l);
                i_idx = i_idx(idx_l); j_idx = j_idx(idx_l); k_idx = k_idx(idx_l);
                d_hat = zeros(length(i_idx),K);
                for j1 = 1 : K
                    d_hat(:,j1) = net_target_ij{l}([X_par{l}(i_idx,:),...
                                repmat(comp_Pij_Y{l}(j1,:),[length(i_idx),1])]')';
                end
                d_hat = d_hat - min(d_hat,[],2);
                num = exp(-(1/T)*d_hat); den = sum(num,2);
                Pij = num./repmat(den,[1 size(num,2)]);
                idx_l = find(k_idx == l);
                loc_cal = [(1:length(i_idx))', j_idx];
                Pij_cal = Pij(sub2ind(size(Pij), loc_cal(:,1), loc_cal(:,2)));
                %Y_parNest = Y_par{l} + mu* V_t;
                %normG = sum(Pij_cal.*(Y_par{l} - X_par{l}(i_idx,:)));
                %normG = (1/length(idx_l))*sum(Pij_cal.*(Y_parNest - X_par{l}(i_idx,:)));
                %V_t = mu*V_t - alpha*normG/(max(1,norm(normG)/tau));
                %Y_par{l} = Y_par{l} + V_t;
                %disp(norm(V_t));
                Grad = (1/length(idx_l))*sum(Pij_cal.*(Y_par{l} - X_par{l}(i_idx,:)));
                Y_par{l} = Y_par{l} - alpha*Grad/norm(Grad);
                disp(norm(alpha*(1/length(idx_l))*sum(Pij_cal.*(Y_par{l} - X_par{l}(i_idx,:)))));
                 
                if norm(Y_old - Y_par{l})/(norm(Y_par{l}) + 1e-12) < epsY
                    count_Y(l) = count_Y(l) + 1;
                else
                    count_Y(l) = 0;
                end
                %if count_Y(l) >= K_consec
                %    break;
                %end
                Y_old = Y_par{l};
            end
        end

        for j = 1 : K
            Y(j,:) = Y_par{j};
        end
        disp(T);
        T = tau1*T;
        scatter(X(:,1),X(:,2),'.'); hold on;
        scatter(Y(:,1),Y(:,2),'d','filled'); title(T); hold off;
        frame = getframe(gcf);       % gcf = current figure
        writeVideo(v, frame);
        count = count + 1;
        if T < Tmin
            break;
        else
            net = init(net);
        end
    end
    

    P_idx = zeros(M,1);
    for i = 1 : M
        d_bar = zeros(K,1);
        for j = 1 : K
            d_bar(j) = net_target([X(i,:),Y(j,:)]');
        end
        d_bar = d_bar-min(d_bar);
        num = exp(-(1/T)*d_bar);
        den = sum(num);
        P(i,:) = num/den;
        [~,P_idx(i)] = min(d_bar);
    end
    
    idx = cell(K,1);
    for k = 1 : K
        idx{k} = find(P_idx==k);
        col = rand(1,3);
        scatter(X(idx{k},1),X(idx{k},2),90,'filled','MarkerEdgeColor','k',...
            'MarkerFaceColor',col,'LineWidth',0.25); hold on;
        scatter(Y(k,1),Y(k,2),500,'p','MarkerEdgeColor', 'k', 'MarkerFaceColor', col, 'LineWidth',2);
    end
    xlim([-8 8]); ylim([-6 6]); axis square; box on;
    set(gca, 'FontSize', 25); set(gca, 'LineWidth', 1.0);
    xticks([-5 0 5]); yticks([-5 0 5]); hold off;
    title("MaxInnerTheta = " + MaxInnerTheta + " MaxInnerY = " + MaxInnerY);
    name = ['SGD_Mod_Run_' num2str(MaxInnerTheta) '_' num2str(MaxInnerY) '.png'];
    print(gcf, name, '-dpng', '-r600');
    name = ['SGD_Mod_Run_Results_' num2str(MaxInnerTheta) '_' num2str(MaxInnerY) '.mat'];
    save(name)
end
toc;
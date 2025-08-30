%% This code is about getting the DQN policy part work well for a fixed Y
%Y = [-3.9640, 2.9850; 3.8190, -2.7215; -4.0105, -2.8831; 4.0636, 2.3994];
clc; clear; close all;
MaxInnerTheta = 200;                % policy update loop max iterations
MaxInner = [500, 250];
inner = 1;
NN_update = 25;

idx = 4; [X,K,T_P,M,N] = data_RLClustering_ModelFree(idx);
[X,mu1,sig] = zscore(X);

T = 0.004; Tmin = 0.004; tau = 0.98;    % annealing parameters
eps = 1;                          % for epsilon greedy policy 
Sz_miniBatch = 256;                 % size of the minibatch for both nets and Y
buf_cap = 1000;                   % memory or replay capacity


H_target = MaxInnerTheta/10;        % steps between target net updates

alpha = 0.001;                      % learning rate for Y SGD
mu = 0.9;
gamma = 0.001;                      % learning rate for policy SGD update

Px = (1/M)*ones(M,1);               % weight for each user data point
[idx_clust, Y] = kmeans(X,K);        % Starting with Y

P = 1/K*ones(M,K);                  % Policy 


idx_input = randi([1 length(X)],[Sz_miniBatch,1]);
init_trainInput = zeros(Sz_miniBatch,N+K+K*N);
init_trainOutput = zeros(Sz_miniBatch,1);
for i = 1 : Sz_miniBatch
    temp = zeros(K,1); temp(idx_clust(idx_input(i)),1) = 1;
    init_trainInput(i,:) = [X(idx_input(i),:) temp' Y(:)'];
    init_trainOutput(i) = norm(X(idx_input(i),:)-Y(idx_clust(idx_input(i)),:))^2;
end
init_trainInput = init_trainInput';
init_trainOutput = init_trainOutput';

%net = feedforwardnet([32 32], 'trainlm');
net = feedforwardnet([10 10], 'trainlm');
net.trainParam.epochs = 250; 
net.trainParam.showWindow = false;
net = train(net, init_trainInput, init_trainOutput);
net.performParam.regularization = 0.1;

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
net = init(net);

%Y = [-2.1694, 1.7170;2.2185, -1.4463; -2.8815, -1.3287; 2.8277, 1.2291];
Y = [-0.1765, 0.0891; 0.1889, 0.0027; 0.3867, -0.0785; -0.2351, -0.2803];
%Y = [2.7705,1.1120; -2.5007, 0.9549; 2.5879, -1.3519; -2.7491, -0.6127];
%Y = [-1.6011, 1.9002; 2.0521, -1.6652; -2.2850, -1.1925; 2.0983, 1.3845];
%Y = [ 1.2128, -0.7145; -1.0914, -0.7233; -1.2498, 0.9193; 0.8504, 0.7206];
%Y = [1.1187, 0.6744; -0.8040, -0.4969;-0.9109, 0.4710; 0.4315, 0.5992];
%Y = [1.1317, -0.4651; -0.5619, -0.2726; -0.3765, 0.5878; -0.1744, 0.6137];
%Y = (Y - mu1)./repmat(sig,[K,1]);

while T >= Tmin
    
    % ===== "policy" loop: update P given current Y =====
    smallCnt = 0;
    for t = 1 : MaxInnerTheta * 2
        i = randi(M); % sampling a user data point

        d_bar = zeros(K,1);
        for j = 1 : K
            %d_bar(j) = nn_forward(theta_target, X(i,:), Y(j,:));
            temp = zeros(K,1); temp(j) = 1;
            d_bar(j) = net_target([X(i,:), temp' Y(:)']');
        end
        d_bar2 = d_bar - min(d_bar);
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
            miniBatchDataX = zeros(Sz_miniBatch,N+K+N*K);
            miniBatchDataY = zeros(Sz_miniBatch,1);
            for b = 1 : Sz_miniBatch
                ii = memory.i(miniBatch(b));
                jj = memory.j(miniBatch(b));
                d_t = memory.d(miniBatch(b));
                temp = zeros(K,1); temp(jj) = 1;
                miniBatchDataX(b,:) = [X(ii,:), temp', Y(:)'];
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
        tau2 = 0.01;
            
            for ii = 1: numel(net.IW)
                net_target.IW{ii} = tau2*net.IW{ii} + (1-tau2)*net_target.IW{ii};
            end

            for ii = 1 : numel(net.LW)
                net_target.LW{ii} = tau2*net.LW{ii} + (1-tau2)*net_target.LW{ii};
            end

            for ii = 1 : numel(net.b)
                net_target.b{ii} = tau2*net.b{ii} + (1-tau2)*net_target.b{ii};
            end
        % if mod(t, 100) == 0
        %     net_target = net;
        % end
        disp(t);
        eps = eps*0.99;
    end
    T = tau*T;
    disp(T);
end
P_idx = zeros(M,1);
for i = 1 : M
    d_bar = zeros(K,1);
    for j = 1 : K
        temp = zeros(K,1); temp(j) = 1;
        d_bar(j) = net_target([X(i,:), temp', Y(:)']');
    end
    d_bar2 = d_bar - min(d_bar);
    num = exp(-(1/T)*d_bar2);
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
xlim([-2 2]); ylim([-2 2]); axis square; box on;
set(gca, 'FontSize', 25); set(gca, 'LineWidth', 1.0);
xticks([-2 0 2]); yticks([-2 0 2]); hold off;
    
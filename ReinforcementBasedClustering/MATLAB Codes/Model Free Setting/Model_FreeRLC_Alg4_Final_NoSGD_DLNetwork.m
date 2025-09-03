%% Algorithm 4: Deep Clustering Algorithm (large N/ continuous X) - No SGD (dlnetwork version)
tic; clear; clc; rng(1);

MaxInnerTheta = 200;                % policy update loop max iterations
MaxInnerY     = 200;                % max iterations in Y update loop
MaxInner      = [500, 50];
NN_update     = 50;

v = VideoWriter('CheckingMiniBatchSGDPerformanceRun2.mp4','Motion JPEG 2000');
v.FrameRate = 1; open(v);

for inner = 1:size(MaxInner,1)

    close all; clc;
    idx = 4; [X,K,T_P,M,N] = data_RLClustering_ModelFree(idx);
    [X,~,~] = zscore(X);

    col = rand(K,3);

    % ----- annealing / algo params -----
    T = 0.5; Tmin = 0.001; tau1 = 0.9;
    eps = 1;                                 % epsilon-greedy
    Sz_miniBatch = 256;                      % minibatch size for both net and Y
    buf_cap = 1000;                          % replay capacity

    H_target = MaxInnerTheta/10;             %#ok<NASGU>  (kept for parity)
    mu = 0.9;                                %#ok<NASGU>  (kept for parity)
    gamma = 1e-3;                            % LR for policy NN (Adam)
    tau = 1;                                 %#ok<NASGU>

    Px = (1/M)*ones(M,1);
    [idx_clust, Y] = kmeans(X,K);            % init Y
    P = 1/K*ones(M,K);                       % policy

    % ----- warm-start data for NN -----
    idx_input = randi([1 length(X)], [Sz_miniBatch,1]);
    INITSZ = Sz_miniBatch;
    IN_DIM = N + K + K*N;  % [x, onehot(j), Y(:)]
    initX  = zeros(INITSZ, IN_DIM, 'single');
    initY  = zeros(INITSZ, 1,       'single');

    for i = 1:INITSZ
        temp = zeros(K,1); temp(idx_clust(idx_input(i))) = 1;
        Y_S = Y + 0.5*randn(K,N);
        initX(i,:) = single([X(idx_input(i),:) temp' Y_S(:)']);
        initY(i)   = single(norm(X(idx_input(i),:) - Y(idx_clust(idx_input(i)),:))^2);
    end
    Y = repmat(Px'*X, [K,1]) + 0.001*randn(K,N);

    % ----- Build dlnetwork regressor (replaces fitnet([15 10], 'trainlm')) -----
    dlnet = buildRegressor(IN_DIM, 1, [15 10]);

    % Warm-up training on init batch (a few epochs)
    dlnet = trainOnArrayBatches(dlnet, initX, initY, ...
        'Epochs', 5, 'BatchSize', Sz_miniBatch, 'LearnRate', 1e-3, 'L2', 0.15);

    % Target network
    dlnet_target = dlnet;

    % ----- Replay memory -----
    memory.i    = zeros(buf_cap,1);
    memory.j    = zeros(buf_cap,1);
    memory.k    = zeros(buf_cap,1);
    memory.d    = zeros(buf_cap,1,'single');
    memory.Yloc = zeros(buf_cap, K*N, 'single');
    buf_n = 0;
    circ_ptr = 1;

    thetaPrev = packLearnables(dlnet);
    K_consec  = 20; %#ok<NASGU>
    epsTheta  = 1e-6;
    epsY      = 1e-4; %#ok<NASGU>

    MaxInnerTheta = MaxInner(inner,1);
    MaxInnerY     = MaxInner(inner,2);
    count = 1;

    while T >= Tmin

        % ===== policy loop: update P given current Y =====
        eps = 0.99;
        smallCnt = 0;
        Yold = Y;

        for t = 1:(MaxInnerTheta*2)
            i = randi(M);                          % sample user
            % Evaluate d_bar(j) for all clusters using target net
            d_bar = predict_dhat_allJ(dlnet_target, X(i,:), Y);

            % softmax over -d_bar/T
            d_bar2 = d_bar - min(d_bar,[],2);
            num = exp(-(1/T)*d_bar2);
            den = sum(num);
            P(i,:) = num/den;

            % epsilon-greedy over P(i,:)
            if rand < eps
                j = randi(K);
            else
                j = randsample(1:K, 1, true, P(i,:));
            end

            % environment transition: sample k ~ T_P(:,j,i)
            k = randsample(1:K, 1, true, T_P(:,j,i));
            dist = single(norm(X(i,:) - Y(k,:))^2);

            % push to replay
            if buf_n < buf_cap
                loc = circ_ptr; buf_n = buf_n + 1;
            else
                loc = circ_ptr;
            end
            circ_ptr = circ_ptr + 1; if circ_ptr > buf_cap, circ_ptr = 1; end

            memory.i(loc)    = i;
            memory.j(loc)    = j;
            memory.k(loc)    = k;
            memory.d(loc)    = dist;
            memory.Yloc(loc,:) = single(Y(:)');

            % train policy NN when enough data
            if buf_n >= Sz_miniBatch && mod(t,25)==0
                miniBatch = randi(buf_n, Sz_miniBatch, 1);
                [miniX, miniY] = packMinibatch(memory, miniBatch, X, K, N);
                dlnet = trainOnArrayBatches(dlnet, miniX, miniY, ...
                    'Epochs', 1, 'BatchSize', Sz_miniBatch, 'LearnRate', gamma, 'L2', 0.15);
            end

            % soft target update
            tau2 = 0.01;
            dlnet_target = softUpdate(dlnet_target, dlnet, tau2);

            % jitter Y during policy updates (as in your code)
            eps = eps * 0.999;
            a = -0.1; b = 0.1;
            Y = Yold + (a + (b-a)*rand(K,2));
        end
        Y = Yold;  % restore

        % ===== Y update loop (No-SGD local minimization that calls NN) =====
        for t = 1:MaxInnerY
            lb2 = Yold - [0.05,0.05];
            ub2 = Yold + [0.05,0.05];
            if buf_n >= 100
                idxb = randperm(buf_n,100);
                Xsub = X(memory.i(idxb),:);
            else
                Xsub = X(randperm(M, min(100,M)), :);
            end
            % IMPORTANT: pass the target network (was net in your code)
            [Y, Fval, ~] = minimize_F_NoSGD(dlnet_target, Xsub, Y, lb2, ub2, T);
            disp(t);
        end

        % ===== recompute soft P via target net for all points =====
        d_hat = zeros(M,K,'single');
        for j1 = 1:K
            d_hat(:,j1) = predict_dhat_fixedJ(dlnet_target, X, Y, j1);
        end
        d_hat = d_hat - min(d_hat,[],2);
        num = exp(-(1/T)*d_hat); den = sum(num,2);
        Pij = num ./ repmat(den,[1 size(num,2)]);

        disp(Y); disp(Fval); disp(T);
        T = tau1*T;

        % ===== Plot & record video frame =====
        col_all = Pij(:,1:K) * col(1:K,:);
        scatter(X(:,1), X(:,2), 90, col_all, 'filled', 'MarkerEdgeColor','k','LineWidth',0.15); hold on;
        for k1 = 1:K
            scatter(Y(k1,1), Y(k1,2), 500, 'p', 'MarkerEdgeColor','k', 'MarkerFaceColor', col(k1,:), 'LineWidth', 2);
        end
        xlim([-2 2]); ylim([-2 2]); axis square; box on;
        set(gca, 'FontSize', 25); set(gca, 'LineWidth', 1.0);
        xticks([-2 0 2]); yticks([-2 0 2]); hold off;
        frame = getframe(gcf); writeVideo(v, frame);

        count = count + 1;
        % No-op: dlnetwork does not need re-init like old 'init(net)'
    end

    % ===== final hard assignment & plot =====
    P_idx = zeros(M,1);
    for i = 1:M
        d_bar = predict_dhat_allJ(dlnet_target, X(i,:), Y);
        d_bar = d_bar - min(d_bar);
        [~, P_idx(i)] = min(d_bar);
    end

    clf; hold on;
    for k1 = 1:K
        idxk = find(P_idx==k1);
        colk = rand(1,3);
        scatter(X(idxk,1), X(idxk,2), 90, 'filled', 'MarkerEdgeColor','k', ...
            'MarkerFaceColor', colk, 'LineWidth', 0.25);
        scatter(Y(k1,1), Y(k1,2), 500, 'p', 'MarkerEdgeColor','k', 'MarkerFaceColor', colk, 'LineWidth', 2);
    end
    xlim([-2 2]); ylim([-2 2]); axis square; box on;
    set(gca, 'FontSize', 25); set(gca, 'LineWidth', 1.0);
    xticks([-2 0 2]); yticks([-2 0 2]); hold off;
    title("MaxInnerTheta = " + MaxInnerTheta + " MaxInnerY = " + MaxInnerY);

    name = ['SGD_Mod_Run_' num2str(MaxInnerTheta) '_' num2str(MaxInnerY) '.png'];
    print(gcf, name, '-dpng', '-r600');
    name = ['SGD_Mod_Run_Results_' num2str(MaxInnerTheta) '_' num2str(MaxInnerY) '.mat'];
    save(name)

end
close(v);
toc;

%% -------------------------- Helpers --------------------------

function dlnet = buildRegressor(inDim, outDim, hidden)
layers = [
    featureInputLayer(inDim, 'Name','in', 'Normalization','none')
    fullyConnectedLayer(hidden(1), 'Name','fc1')
    reluLayer('Name','r1')
    fullyConnectedLayer(hidden(2), 'Name','fc2')
    reluLayer('Name','r2')
    fullyConnectedLayer(outDim, 'Name','out') % linear head (regression)
];
lgraph = layerGraph(layers);
dlnet  = dlnetwork(lgraph);
end

function [loss, grads] = modelGradients(dlnet, dlX, dlT, L2)
dlY   = forward(dlnet, dlX);        % [1 x B]
mseL  = mean((dlY - dlT).^2, 'all');

if L2 > 0
    W = dlnet.Learnables;
    l2sum = 0;
    for i = 1:height(W)
        if contains(string(W.Parameter(i)), "Weights")
            l2sum = l2sum + sum(W.Value{i}.^2,'all');
        end
    end
    mseL = mseL + L2 * l2sum;
end

loss  = mseL;
grads = dlgradient(loss, dlnet.Learnables);
end

function dlnet = trainOnArrayBatches(dlnet, X, Y, varargin)
% X: [B x Din], Y: [B x 1]
p = inputParser;
p.addParameter('Epochs', 1);
p.addParameter('BatchSize', 256);
p.addParameter('LearnRate', 1e-3);
p.addParameter('L2', 0.0);
p.parse(varargin{:});
opt = p.Results;

B = size(X,1);
beta1 = 0.9; beta2 = 0.999; eps = 1e-8;
trailingAvg = []; trailingAvgSq = [];

for e = 1:opt.Epochs
    idx = randperm(B);
    for s = 1:opt.BatchSize:B
        eidx = min(s+opt.BatchSize-1, B);
        ib = idx(s:eidx);

        Xb = X(ib,:).';
        Yb = Y(ib,:).';

        dlX = dlarray(Xb, 'CB');  % [Din x b]
        dlT = dlarray(Yb, 'CB');  % [1   x b]
        if canUseGPU, dlX = gpuArray(dlX); dlT = gpuArray(dlT); end

        [loss, grads] = dlfeval(@modelGradients, dlnet, dlX, dlT, opt.L2);
        [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, grads, ...
            trailingAvg, trailingAvgSq, (e*1e7 + s), opt.LearnRate, beta1, beta2, eps); %#ok<NASGU>
    end
end
end

function dlnetTarget = softUpdate(dlnetTarget, dlnet, tau)
% dlnetTarget = tau * dlnet + (1-tau) * dlnetTarget
LT = dlnetTarget.Learnables;
L  = dlnet.Learnables;
for i = 1:height(L)
    if isnumeric(L.Value{i})
        LT.Value{i} = tau*L.Value{i} + (1-tau)*LT.Value{i};
    else
        % leave non-numeric state (rare)
    end
end
dlnetTarget = dlnetwork(layerGraph(dlnetTarget));  % refresh internal state
dlnetTarget = assignLearnables(dlnetTarget, LT);
end

function net = assignLearnables(net, tbl)
% Replace Learnables table (utility)
net = setLearnables(net, tbl);
end

function vec = packLearnables(dlnet)
L = dlnet.Learnables;
chunks = cell(height(L),1);
for i = 1:height(L)
    v = L.Value{i};
    if isnumeric(v)
        chunks{i} = gather(v(:));
    else
        chunks{i} = zeros(0,1,'single');
    end
end
vec = single(cat(1, chunks{:}));
end

function [miniX, miniY] = packMinibatch(memory, idxs, X, K, N)
B = numel(idxs);
IN_DIM = N + K + K*N;
miniX = zeros(B, IN_DIM, 'single');
miniY = zeros(B, 1,      'single');
for b = 1:B
    m  = idxs(b);
    ii = memory.i(m);
    jj = memory.j(m);
    d_t = memory.d(m);
    Y_S = memory.Yloc(m,:);     % [1 x (K*N)]

    temp = zeros(K,1); temp(jj) = 1;
    miniX(b,:) = single([X(ii,:), temp', Y_S(:)']);
    miniY(b)   = single(d_t);
end
end

function dbar = predict_dhat_allJ(dlnet, x_i, Y)
% returns [K x 1] distances for fixed x_i, all j, current Y
K = size(Y,1); N = size(Y,2);
IN_DIM = numel(x_i) + K + K*N;
B = K;
AA = zeros(IN_DIM, B, 'single');
xrow = single(x_i(:)).';
Yvec = single(Y(:)).';
for j = 1:K
    onehot = zeros(1,K,'single'); onehot(j) = 1;
    AA(:,j) = single([xrow, onehot, Yvec]).';
end
dlX = dlarray(AA, 'CB'); if canUseGPU, dlX = gpuArray(dlX); end
dlY = forward(dlnet, dlX); y = gather(extractdata(dlY));  % [1 x K]
dbar = y(:);
end

function ycol = predict_dhat_fixedJ(dlnet, X, Y, j)
% vectorized over all i, fixed j
[M, N] = size(X); K = size(Y,1);
IN_DIM = N + K + K*N;
B = M;
AA = zeros(IN_DIM, B, 'single');
onehot = zeros(1,K,'single'); onehot(j) = 1;
Yvec = single(Y(:)).';
for i = 1:M
    AA(:,i) = single([X(i,:), onehot, Yvec]).';
end
dlX = dlarray(AA, 'CB'); if canUseGPU, dlX = gpuArray(dlX); end
dlY = forward(dlnet, dlX);
ycol = gather(extractdata(dlY)).';   % [M x 1]
end

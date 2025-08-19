%% Algorithm 4: Deep Clustering Algorithm (large N/ continuous X)
clear; clc; rng(1);

% ------------------ initialization -------------------
tic; close all; clc; 
idx = 4; [X,K,T_P,M,N] = data_RLClustering_ModelFree(idx);
[X,mu,sig] = zscore(X);

T = 5; Tmin = 0.02; tau = 0.98;    % annealing parameters
eps = 0.1;                          % for epsilon greedy policy 
H_target = 250;                     % steps between target net updates
Sz_miniBatch = 512;                 % size of the minibatch for both nets and Y
buf_cap = 200000;                   % memory or replay capacity

MaxInnerTheta = 1500;               % policy update loop max iterations
MaxInnerY = 1500;                   % max iterations in Y update loop

alpha = 0.005;                      % learning rate for Y SGD
gamma = 0.001;                      % learning rate for policy SGD update

Px = (1/M)*ones(M,1);               % weight for each user data point
Y = repmat(Px'*X, [K,1]);           % Starting with Y
P = 1/K*ones(M,K);                  % Policy 

theta = nn_init([2*N 32 16 1]);     % ANN approximator for dbar(i,j)
theta_target = theta;

memory.i = zeros(buf_cap,1); memory.j = zeros(buf_cap,1);
memory.k = zeros(buf_cap,1); memory.d = zeros(buf_cap,1);


buf_n = 0;                          % current size of the memory
circ_ptr = 1;                       % circular pointer

while T >= Tmin
    
    % ===== "policy" loop: update P given current Y =====
    for t = 1 : MaxInnerTheta
        i = randi(M); % sampling a user data point

        d_bar = zeros(K,1);
        for j = 1 : K
            d_bar(j) = nn_forward(theta_target, X(i,:), Y(j,:));
        end
        P(i,:) = (softmax_row(-(1/T)*d_bar))';

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
        memory.i(loc) = i; memory.j(loc) = j; 
        memory.k(loc) = k; memory.d(loc) = dist;
        circ_ptr = circ_ptr + 1; 
        if circ_ptr > buf_cap
            circ_ptr = 1;
        end
        
        % train theta when enough data is available
        if buf_n >= Sz_miniBatch
            miniBatch = randi(buf_n, Sz_miniBatch, 1);   % uniformly sample
            loss = 0; gsum = nn_zeros_like(theta);  % what is this function?
            for b = 1 : Sz_miniBatch
                ii = memory.i(miniBatch(b));
                jj = memory.j(miniBatch(b));
                d_t = memory.d(miniBatch(b));
                % forward and backward propagation on (x_i, y_j)
                [d_pred, cache] = nn_forward(theta, X(ii,:), Y(jj,:),true); % what is cache here?
                [g, ~] = nn_backward(theta, cache, d_pred - d_t); % computing gradient?
                gsum = nn_add(gsum, g);
                loss = loss + 0.5*(d_pred - d_t)^2;
            end
            [theta, adam_state] = nn_adam(theta, gsum, Sz_miniBatch, gamma); % what is adma_state?
        end

        % target network sunc
        if mod(t, H_target) == 0
            theta_target = theta;
        end
    end

    % ===== "centroid" loop: update Y given current P and stochasticity =====
    
    Theta_ij = cell(K,1); Par_memory = cell(K,1); 
    X_par = cell(K,1); Y_par = cell(K,1);
    for k = 1 : K
        Theta_ij{k} = theta_target; X_par{k} = X; Y_par{k} = Y;
        Par_memory{k} = memory;
    end
    
    parfor l = 1 : K
        for t = 1 : MaxInnerY
            idx = randi([1 buf_n]);
            i = Par_memory{l}.i(idx); j = Par_memory{l}.j(idx); k = Par_memory{l}.k(idx);
            d_hat = zeros(1,K);
            for j1 = 1 : K
                d_hat(j1) = nn_forward(theta_target, X_par{l}(i,:), Y_par{l}(j1,:));
            end
            Pij = softmax_row(-(1/T)*d_hat);
            if k == l
                Y(l,:) = Y(l,:) - alpha*Pij(j)*(Y(l,:) - X_par{l}(i,:));
            end
        end
    end

    disp(T);
    T = tau*T;
end

%% functions required in the above script

function s = softmax_row(z)
    z = z - max(z,[],2);
    e = exp(z);
    s = e/sum(e);
end

% ----------------- neural net: manual MLP ---------------- 
function theta = nn_init(sizes)
    % sizes: [in, h1, h2, out]
    theta.W1 = 0.1*randn(sizes(1),sizes(2));
    theta.b1 = zeros(1,sizes(2));
    theta.W2 = 0.1*randn(sizes(2),sizes(3));
    theta.b2 = zeros(1,sizes(3));
    theta.W3 = 0.1*randn(sizes(3),sizes(4));
    theta.b3 = zeros(1,sizes(4));
    theta.adam_m = nn_zeros_like(theta);
    theta.adam_v = nn_zeros_like(theta);
    theta.adam_t = 0;
end

function Z = nn_cat(x,y)
    Z = [x(:).', y(:).'];
end

function [y, cache] = nn_forward(theta, x, yj, want_cache)
    a0 = nn_cat(x,yj);
    z1 = a0*theta.W1 + theta.b1; h1 = max(0,z1);  % ReLU activation function
    z2 = h1*theta.W2 + theta.b2; h2 = max(0,z2);  % ReLU
    y = h2*theta.W3 + theta.b3;                   % scalar output
    if nargin>3 && want_cache
        cache.a0=a0; cache.z1=z1; cache.h1=h1; cache.z2=z2; cache.h2=h2;
    end
end

function [g, dy_da0] = nn_backward(theta, cache, dl_dy)
    % dL/dy provided. Backpropagation through ReLU layers
    dh2 = dl_dy* theta.W3';
    dz2 = dh2; dz2(cache.z2<=0)=0;
    dW3 = (cache.h2')*dl_dy; db3 = dl_dy;

    dh1 = dz2*theta.W2';
    dz1 = dh1; dz1(cache.z1<=0)=0;
    dW2 = (cache.h1')*dz2; db2 = dz2;

    da0 = dz1*theta.W1';  % what is this?
    dW1 = (cache.a0')*dz1; db1 = dz1;

    g.W1=dW1; g.b1=db1; g.W2=dW2; g.b2=db2; g.W3=dW3; g.b3=db3;
    dy_da0 = []; % what is this?
end

function z = nn_zeros_like(theta)
    z.W1 = zeros(size(theta.W1)); z.b1 = zeros(size(theta.b1));
    z.W2 = zeros(size(theta.W2)); z.b2 = zeros(size(theta.b2));
    z.W3 = zeros(size(theta.W3)); z.b3 = zeros(size(theta.b3));
end

function sumg = nn_add(a,b)
    sumg.W1 = a.W1 + b.W1; sumg.b1 = a.b1 + b.b1;
    sumg.W2 = a.W2 + b.W2; sumg.b2 = a.b2 + b.b2;
    sumg.W3 = a.W3 + b.W3; sumg.b3 = a.b3 + b.b3;
end

function [theta, state] = nn_adam(theta, gsum, B, lr)
    % one Adam step on averages (gsum/B)
    if ~isfield(theta,'adam_t'), theta.adam_t = 0; end % what does this do?
    beta1 = 0.9; beta2 = 0.999; eps=1e-08;
    theta.adam_t = theta.adam_t + 1;

    fields = {'W1','b1','W2','b2','W3','b3'};
    for f = 1 : numel(fields)
        F = fields{f};
        g = gsum.(F)/B;

        theta.adam_m.(F) = beta1*theta.adam_m.(F) + (1-beta1)*g;
        theta.adam_v.(F) = beta2*theta.adam_v.(F) + (1-beta2)*(g.^2);
        mhat = theta.adam_m.(F) / (1 - beta1^theta.adam_t);
        vhat = theta.adam_v.(F) / (1 - beta2^theta.adam_t);

        theta.(F) = theta.(F) - lr * mhat ./ (sqrt(vhat) + eps);
    end
    state = [];
end
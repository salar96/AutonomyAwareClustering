function [Y_opt, Fval, C] = minimize_F_NoSGDv2(net, X, Y0, lb, ub, T)
% MINIMIZE_F_NOSGD  (dlnetwork-compatible, fmincon-safe doubles)
% net : dlnetwork  | X:[M x N] | Y0,lb,ub:[K x N] | T:scalar

    [~, N] = size(X);
    K      = size(Y0,1);

    % Ensure optimizer variables are double
    Y0 = double(Y0); lb = double(lb); ub = double(ub);

    function F = objFun(yvec)
        % yvec will be double; keep the math in double
        Y = reshape(yvec, K, N);
        C_local = pairwise_cost_matrix(net, X, Y);   % returns double [M x K]
        Cshift  = C_local - min(C_local, [], 2);     % double
        % Your original objective, but ensure final F is double
        F = -T * sum( -(1/T)*min(C_local,[],2) + log(sum(exp(-(1/T)*Cshift), 2)) );
        F = double(F);                               % <-- critical for fmincon
    end

<<<<<<< HEAD
    opts = optimoptions('fmincon','Display','iter-detailed','MaxIterations',1);
    
    [yopt, Fval] = fmincon(@objFun,Y0,[],[],[],[],lb,ub,[],opts);
    %[yopt, Fval] = fminunc(@objFun,Y0,opts);
    Y_opt = reshape(yopt,K,N);
=======
    opts = optimoptions('fmincon', 'Display','iter-detailed');

    [yopt, Fval] = fmincon(@objFun, Y0, [], [], [], [], lb, ub, [], opts);
    Y_opt = reshape(yopt, K, N);

    % Return C at optimum (double)
    C = pairwise_cost_matrix(net, X, Y_opt);
>>>>>>> 82085b0 (files and results)
end

function C = pairwise_cost_matrix(net, X, Y)
% C(i,j) = predicted cost for x_i vs centroid j
% Build inputs in single for the network, but return DOUBLE to the optimizer.
    [M, N] = size(X);
    K      = size(Y,1);

    % Big input batch: [X_tile, onehot(j), vec(Y)] for all (i,j)
    X_tile  = kron(ones(K,1), X);            % [K*M x N] (double)
    onehot  = kron(eye(K), ones(M,1));       % [K*M x K] (double)
    Yvec    = Y(:)';                         % [1 x (K*N)] (double)
    Y_block = repmat(Yvec, [K*M, 1]);        % [K*M x (K*N)] (double)

    % The network is happy with single, so cast only for the forward pass
    Inp_single = single([X_tile, onehot, Y_block]);   % [K*M x Din]
    dlX = dlarray(Inp_single', 'CB');                 % [Din x (K*M)]
    if canUseGPU, dlX = gpuArray(dlX); end

    dlY  = forward(net, dlX);                         % [1 x (K*M)], usually single
    dist = gather(extractdata(dlY));                  % numeric (single)
    dist = double(dist);                              % <-- make it double for fmincon
    C    = reshape(dist, M, K);                       % [M x K] (double)
end

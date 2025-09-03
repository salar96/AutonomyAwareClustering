function [Y_opt, Fval, C, exitflag, output] = minimize_F_with_net_box(net, X, Pi, Y0, lbN, ubN, T)
% Minimize F = sum_{i=1}^M sum_{j=1}^K Pi(i,j) * net([x_i, onehot(j), vec(Y)])
% subject to elementwise bounds lbN <= Y <= ubN.
%
% net : trained net taking (N + K + K*N) x S inputs -> 1 x S outputs
% X   : M x N   inputs x_i
% Pi  : M x K   weights π(j|i) (rows sum ~ 1)
% Y0  : K x N   initial centers
% lbN : 1 x N or K x N lower bounds for Y
% ubN : 1 x N or K x N upper bounds for Yal point is a local minimum that satisfies the constraints.

    % ---- shape checks ----
    [M,N]   = size(X);
    [M2,K]  = size(Pi);  assert(M2==M,'Pi must be MxK');
    [K2,N2] = size(Y0);  assert(K2==K && N2==N,'Y0 must be KxN');

    % normalize Pi rows (tolerant)
    rs = sum(Pi,2);  Pi = Pi ./ max(rs, eps);

    % broadcast bounds if given as 1xN
    if isvector(lbN) && numel(lbN)==N, lbN = repmat(lbN(:).', K, 1); end
    if isvector(ubN) && numel(ubN)==N, ubN = repmat(ubN(:).', K, 1); end
    assert(isequal(size(lbN),[K,N]) && isequal(size(ubN),[K,N]), 'lbN/ubN must be 1xN or KxN');

    % flatten for fmincon
    lb = lbN(:);  ub = ubN(:);  y0 = Y0(:);

    % objective
    function F = objFun(yvec)
        Y = reshape(yvec, K, N);
        C = pairwise_cost_matrix_new(net, X, Y);   % M x K, C(i,j)=net([x_i, e_j, vec(Y)])
        F = -T*sum(-(1/T)*min(C,[],2) + log(sum(exp(-(1/T)*(C-min(C,[],2))),2)));
    end

    % solver
    opts = optimoptions('fmincon','Display','iter', ...
        'FiniteDifferenceType','central', ...
        'StepTolerance',1e-9, ...
        'OptimalityTolerance',1e-10, ...
        'MaxIterations',500);

    [yopt, Fval, exitflag, output] = fmincon(@objFun, y0, [],[],[],[], lb, ub, [], opts);
    Y_opt = reshape(yopt, K, N);
end

function C = pairwise_cost_matrix_new(net, X, Y)
% Build inputs for all (i,j): [x_i, onehot(j), vec(Y)], then eval net.
% Returns M x K matrix of costs.

    [M,N] = size(X);
    [K,~] = size(Y);

    % Repeat X for each cluster j → (K*M) x N
    X_tile = kron(ones(K,1), X);

    % One-hot(j): rows 1..M→e1, next M→e2, ... → (K*M) x K
    onehot = kron(eye(K), ones(M,1));

    % vec(Y) repeated for all rows → (K*M) x (K*N)
    Yvec   = Y(:).';                 % 1 x (K*N)
    Y_block= repmat(Yvec, K*M, 1);

    % Concatenate and forward (features x samples)
    IN   = [X_tile, onehot, Y_block];     % (K*M) x (N+K+K*N)
    dval = net(IN.');                      % 1 x (K*M)
    C    = reshape(dval, M, K);            % back to M x K
end

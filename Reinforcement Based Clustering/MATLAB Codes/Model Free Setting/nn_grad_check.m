function flag = nn_grad_check(net,X,Y,J,l)

    [M, N] = size(X);
    K      = size(Y,1);
    assert(l>=1 && l<=K, 'l must be in 1..K');

    % --- extract chain parameters ---
    L  = numel(net.layers);
    W  = cell(L,1); b = cell(L,1); tf = cell(L,1);
    W{1} = net.IW{1,1};          b{1} = net.b{1};          tf{1} = net.layers{1}.transferFcn;
    for t = 2:L
        W{t} = net.LW{t,t-1};    b{t} = net.b{t};          tf{t} = net.layers{t}.transferFcn;
    end

    % --- input layout sizes ---
    Din_x  = N;
    Din_1h = K;
    Din_Y  = K*N;
    y_start = Din_x + Din_1h + 1;
    y_end   = y_start + Din_Y - 1;

    % IMPORTANT: MATLAB column-major mapping for row l in Y(:)
    % indices of Y(l,1), Y(l,2), ..., Y(l,N) inside Y(:):
    idxYl = (l : K : K*N).';         % [N x 1] stride-K positions

    y   = zeros(M,1);
    dYl = zeros(M,N);

    Yvec = Y(:);

    for i = 1:M
        j = J(i);
        % input u = [x_i; onehot(j); vec(Y)]
        onehot = zeros(K,1); onehot(j) = 1;
        u = [X(i,:).'; onehot; Yvec];   % [Din_x+Din_1h+Din_Y x 1]

        % forward
        a = cell(L+1,1); n = cell(L,1);
        a{1} = u;
        for t = 1:L
            n{t}   = W{t}*a{t} + b{t};
            a{t+1} = local_act(tf{t}, n{t});
        end
        y(i) = a{L+1};

        % backprop sensitivities
        s = cell(L,1);
        s{L} = local_act_deriv(tf{L}, n{L}, a{L+1});
        for t = L-1:-1:1
            s{t} = (W{t+1}.' * s{t+1}) .* local_act_deriv(tf{t}, n{t}, a{t+1});
        end

        % gradient wrt input u
        g_u = W{1}.' * s{1};              % [Din_x+Din_1h+Din_Y x 1]

        % take only Y part, then pick row-l entries using stride-K indices
        gY_full = g_u(y_start:y_end);     % [K*N x 1] in column-major order
        dYl(i,:)= (gY_full(idxYl)).';     % [1 x N]
    end
    flag = 0;
    disp([dYl 2*(Y(l,:) - X)]);
end

function a = local_act(name, n)
    switch lower(name)
        case {'purelin'}, a = n;
        case {'tansig'},  a = tansig(n);
        case {'logsig'},  a = logsig(n);
        case {'poslin','relu'}, a = max(0,n);
        case {'satlin'},  a = max(0, min(1, n));
        otherwise, error('Unsupported transferFcn: %s', name);
    end
end

function dp = local_act_deriv(name, n, a)
    switch lower(name)
        case {'purelin'}, dp = ones(size(n));
        case {'tansig'},  dp = 1 - a.^2;
        case {'logsig'},  dp = a .* (1 - a);
        case {'poslin','relu'}, dp = double(n > 0);
        case {'satlin'},  dp = double(n > 0 & n < 1);
        otherwise, error('Unsupported transferFcn: %s', name);
    end
end
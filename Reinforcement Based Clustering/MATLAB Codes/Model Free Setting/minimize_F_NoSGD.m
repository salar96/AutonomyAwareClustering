function [Y_opt, Fval, C] = minimize_F_NoSGD(net,X,Y0,lb,ub,T)

    [~,N] = size(X);
    K = size(Y0,1);
    
    function F = objFun(yvec)
        Y = reshape(yvec,K,N);
        C = pairwise_cost_matrix(net,X,Y);
        F = -T*sum(-(1/T)*min(C,[],2) + log(sum(exp(-(1/T)*(C-min(C,[],2))),2)));
    end

    opts = optimoptions('fmincon','Display','iter-detailed');
    
    [yopt, Fval] = fmincon(@objFun,Y0,[],[],[],[],lb,ub,[],opts);
    Y_opt = reshape(yopt,K,N);
end

function C = pairwise_cost_matrix(net,X,Y)
    [M,~] = size(X);
    K = size(Y,1);

    X_tile = kron(ones(K,1), X);
    onehot = kron(eye(K),ones(M,1));
    Yvec = Y(:)';
    Y_block = repmat(Yvec, [K*M, 1]);

    Inp = [X_tile, onehot, Y_block];
    dist = net(Inp');
    C = reshape(dist, M, K);
end
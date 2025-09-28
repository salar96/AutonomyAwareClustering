function [T_cr, Delta] = critical_beta_NewDelta(X, Y, K, M, T_P, P, rho)

    % Following matrices need to be constructed P_K, P_A, z_i
    Xi = zeros(K*2,M);
    z = zeros(size(Xi));
    for i = 1 : M
        Xi(:,i) = kron(ones(K,1), X(i,:)');
    end
    YY = Y'; z = repmat(YY(:), [1 M]) - Xi; 
    P_hat = cell(M,K); P_hatA = cell(M,K); 
    PP = cell(M,K); PP_A = cell(M,K);
    for i = 1 : M
        for j = 1 : K
            P_hat{i,j} = diag(T_P(:,j,i));
            P_hatA{i,j} = rho(i)*P(i,j)*P_hat{i,j};
            PP{i,j} = kron(P_hat{i,j}, eye(2));
            PP_A{i,j} = kron(P_hatA{i,j}, eye(2));
        end
    end
    P_hatK = zeros(K,K); P_K = zeros(2*K,2*K);
    for k = 1 : K
        P_hatK(k,k) = sum(rho'*(P.*squeeze(T_P(k,:,:))'));
    end
    P_K = kron(P_hatK,eye(2));

    Delta = zeros(2*k,2*k);
    for i = 1 : M
        for j = 1 : K
            Delta = Delta + inv(sqrtm(P_K))*PP_A{i,j}*(z(:,i)*z(:,i)')*PP{i,j}*inv(sqrtm(P_K));
        end
    end
    
    Delta2 = zeros(size(Delta)); PPP = cell(M,1);
    for i = 1 : M
        PPP{i} = zeros(size(PP{i,1}));
        for j = 1 : K
            PPP{i} = PPP{i} + P(i,j)*PP{i,j};
        end
        Delta2 = Delta2 + rho(i)*inv(sqrtm(P_K))*(PPP{i}*z(:,i)*z(:,i)'*PPP{i})*inv(sqrtm(P_K));
    end


    T_cr = 2*max(eig(Delta-Delta2));
    Delta = Delta - Delta2;
    disp(min(eig(Delta)));
end
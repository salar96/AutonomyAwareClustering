function [PSI, f_val] = CrTempRL_Clustering_Computational_v3(X, K, T_P)
    M = size(X,1); N = 2; Tmin = 0.0005; alpha = 0.99; PERTURB = 0.01; 
    STOP = 1e-2; T = 5.2; Px = (1/M)*ones(M,1); Y = repmat(Px'*X, [K,1]);
    PSI = zeros(size(Y)); rho = Px;
    while T >= Tmin
        L_old = inf;

        while 1
            [D,~] = distortion_RLClustering(X,Y,M,N,K,T_P);
            num = exp(-D/T);
            den = repmat(sum(num,2),[1 K]);
            P = num./den;
            Py = P'*Px;
            for l = 1:K
                T_slice = squeeze(T_P(l, :, :))';   
                W = (1/M) * P .* T_slice;           
                row_weights = sum(W, 2);            
                numerator = row_weights' * X;
                denominator = sum(row_weights);
                Y(l, :) = numerator / denominator;
            end
            Y = Y + PERTURB*rand(size(Y));
            if isnan(Y)
                pp = 1;
            end
            L = -T*Px'*log(sum(exp(-D/T),2));
            if(norm(L-L_old) < STOP)
                break;
            end
            L_old = L;
        end

        x_temp = PSI'; x0 = x_temp(:);
        fun = @(x)objective(x);
        nonclon = @(x)nonlin_const(x,K,M,P,Px,X,Y,T_P,T);
        A = []; b = []; Aeq = []; beq = []; lb = -ones(size(x0)); ub = ones(size(x0));
        options = optimoptions('fmincon','Display','off','StepTolerance',1e-04,...
            'MaxFunctionEvaluations',1e03);
        [PSI, f_val] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonclon,options);
        [v1, ~] = nonlin_const(PSI,K,M,P,Px,X,Y,T_P,T);
        fprintf('%d %d \n',T, v1);
        disp(Y);
        %disp(nonclon);
        %[~, Delta] = critical_beta(X, Y, K, M, T_P, P, rho);
        %disp(min(eig(eye(2*K)-2/T*Delta)));
        T = T*alpha;
    end
end
function fun = objective(x)
    fun = 1;
end

% function [c, c_eq] = nonlin_const(x, K, M, P, rho, X, Y, T_P)
%     c = [];
%     c_eq = zeros(2,1);
%     Px = rho;
% 
%     % Reshape and transpose x
%     Psi = reshape(x, [2, K])';  % K×2
%     PSI = x;
% 
%     % Precompute difference tensor: Diff(i,j,:) = X(i,:) - Y(j,:)
%     % Result: M×K×2
%     Diff = permute(X, [1 3 2]) - permute(Y, [3 1 2]);
% 
%     % Precompute dot product: K×M×K
%     % temp_cov(k,j,i) = dot(Psi(k,:), X(i,:) - Y(j,:))
%     temp_cov = zeros(K, K, M);
%     for i = 1:M
%         for k = 1:K
%             temp_cov(k,:,i) = Psi(k,:) * squeeze(Diff(i,:,:))';  % 1×K
%         end
%     end
% 
%     % Sum over k: sum_k T_P(k,j,i) * temp_cov(k,j,i)
%     % Loop over i
%     for i = 1:M
%         weighted_sum = sum(squeeze(T_P(:,:,i)) .* squeeze(temp_cov(:,:,i)), 1);  % 1×K
%         Cov2_i = P(i,:) * weighted_sum';  % scalar
%         c_eq(1) = c_eq(1) + Px(i) * Cov2_i^2;
%     end
% 
%     % Second constraint: norm of PSI = 2
%     c_eq(2) = norm(PSI) - 2;
% end


function [c, c_eq] = nonlin_const(x,K,M,P,rho,X,Y,T_P,T)
    c_eq = zeros(2,1); c = []; Px = rho;
    x_temp = reshape(x,[2,K]); Psi = x_temp';
    PSI = x;
    Cov2 = {};
    for i = 1 : M
        Cov2{i} = 0;
        for j = 1 : K
            temp_cov = 0;
            for k = 1 : K
                temp_cov = temp_cov + T_P(k,j,i)*Psi(k,:)*(X(i,:)-Y(k,:))';
            end
            Cov2{i} = Cov2{i} + P(i,j)*temp_cov;
        end
        c_eq(1) = c_eq(1) + Px(i)*Cov2{i}^2;
    end
    c_eq(2) = norm(PSI) - 2;

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

    Delta = zeros(2*K,2*K);
    for i = 1 : M
        for j = 1 : K
            Delta = Delta + inv(sqrtm(P_K))*PP_A{i,j}*(z(:,i)*z(:,i)')*PP{i,j}*inv(sqrtm(P_K));
        end
    end

    c = PSI'*sqrtm(P_K)*(eye(2*K) - (2/T)*Delta)*sqrtm(P_K)*PSI + c_eq(1);
    c_eq = c_eq(2);
end
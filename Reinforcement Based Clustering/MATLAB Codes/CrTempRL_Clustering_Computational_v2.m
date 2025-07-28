function [PSI, f_val] = CrTempRL_Clustering_Computational_v2(X, K, T_P)
    M = size(X,1); N = 2; Tmin = 0.0005; alpha = 0.99; PERTURB = 0.0001; 
    STOP = 1e-2; T = 9; Px = (1/M)*ones(M,1); Y = repmat(Px'*X, [K,1]);
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
        fun = @(x)objective(x,X,Y,K,M,T_P,P,rho,T);
        nonclon = @(x)nonlin_const(x);
        A = []; b = []; Aeq = []; beq = []; lb = []; ub =[];
        options = optimoptions('fmincon','Display','off','StepTolerance',1e-10);
        [PSI, f_val] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonclon,options);
        fprintf('%d %d \n',T, f_val);
        T = T*alpha;
    end
end
function fun = objective(x,X,Y,K,M,T_P,P,rho,T)
    
    PSI = x; x_temp = reshape(x,[2,K]); Psi = x_temp';
    Xi = zeros(K*2,M); Px = rho;
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
    c_eq = 0;
    Cov2 = cell(M,1);
    for i = 1 : M
        Cov2{i} = 0;
        for j = 1 : K
            temp_cov = 0;
            for k = 1 : K
                temp_cov = temp_cov + T_P(k,j,i)*Psi(k,:)*(X(i,:)-Y(k,:))';
            end
            Cov2{i} = Cov2{i} + P(i,j)*temp_cov;
        end
        c_eq = c_eq + Px(i)*Cov2{i}^2;
    end

    fun = PSI'*sqrtm(P_K)*(eye(2*K) - (2/T)*Delta)*sqrtm(P_K)*PSI + (2/T)*c_eq;
    
end
function [c, c_eq] = nonlin_const(x)
    c = []; c_eq = norm(x) - 1;
end
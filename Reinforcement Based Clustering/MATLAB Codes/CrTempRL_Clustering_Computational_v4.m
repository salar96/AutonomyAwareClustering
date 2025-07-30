function [PSI, f_val] = CrTempRL_Clustering_Computational_v4(X, K, T_P)
    M = size(X,1); N = 2; Tmin = 0.0005; alpha = 0.99; PERTURB = 0.0001; 
    STOP = 1e-2; T = 0.5; Px = (1/M)*ones(M,1); Y = repmat(Px'*X, [K,1]);
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
        A = []; b = []; Aeq = []; beq = [];
        options = optimoptions('fmincon','Display','off','StepTolerance',1e-05,'MaxFunctionEvaluations',1e03);
        [PSI, f_val] = fmincon(fun,x0,A,b,Aeq,beq,[],[],nonclon,options);
        [T_cr, Delta] = critical_beta_NewDelta(X, Y, K, M, T_P, P, rho);
        fprintf('%d %d %d \n',T, T_cr, f_val);
        %disp(Y);
        disp(min(eig(eye(2*K)-2/T*Delta)));
        T = T*alpha;
    end
end
function fun = objective(x,X,Y,K,M,T_P,P,rho,T)
    
    PSI = x;
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

    Delta2 = zeros(size(Delta)); PPP = cell(M,1);
    for i = 1 : M
        PPP{i} = zeros(size(PP{i,1}));
        for j = 1 : K
            PPP{i} = PPP{i} + P(i,j)*PP{i,j};
        end
        Delta2 = Delta2 + rho(i)*inv(sqrtm(P_K))*(PPP{i}*z(:,i)*z(:,i)'*PPP{i})*inv(sqrtm(P_K));
    end
    
    Delta = Delta - Delta2;
    fun = PSI'*sqrtm(P_K)*(eye(2*K) - (2/T)*Delta)*sqrtm(P_K)*PSI;

end

function [c, c_eq] = nonlin_const(x)
    c_eq = []; PSI = x;
    c = -norm(PSI) + 1;
    %c_eq = norm(PSI) - 2;
end
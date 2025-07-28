function [PSI, f_val] = CrTempRL_Clustering_Computational_non_vector(X, K, T_P)
    M = size(X,1); N = 2; Tmin = 0.0005; alpha = 0.99; PERTURB = 0.0001; 
    STOP = 1e-2; T = 8.7; Px = (1/M)*ones(M,1); Y = repmat(Px'*X, [K,1]);
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
        nonclon = @(x)nonlin_const(x,K,M,P,Px,X,Y,T_P);
        A = []; b = []; Aeq = []; beq = []; lb = -ones(size(x0)); ub = ones(size(x0));
        options = optimoptions('fmincon','Display','off','StepTolerance',1e-05);
        [PSI, f_val] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonclon,options);
        fprintf('%d %d \n',T, f_val);
        T = T*alpha;
    end
end
function fun = objective(x,X,Y,K,M,T_P,P,rho,T)
    
    Px = rho; x_temp = reshape(x,[2,K]); Psi = x_temp';
    Cov2 = cell(M,2); fun1 = 0; fun2 = 0;
    for i = 1 : M
        Cov2{i,1} = 0; Cov2{i,2} = 0;
        for j = 1 : K
            temp_cov = 0; temp_cov2 = 0;
            for k = 1 : K
                temp_cov = temp_cov + T_P(k,j,i)*Psi(k,:)*(X(i,:)-Y(k,:))';
                temp_cov2 = temp_cov2 + T_P(k,j,i)*Psi(k,:)*Psi(k,:)';
            end
            Cov2{i,1} = Cov2{i,1} + P(i,j)*temp_cov^2;
            Cov2{i,2} = Cov2{i,2} + P(i,j)*temp_cov2;
        end
        fun1 = fun1 + Px(i)*Cov2{i,1};
        fun2 = fun2 + Px(i)*Cov2{i,2};
    end

    fun = (-2/T)*fun1 + fun2;
    
end

function [c, c_eq] = nonlin_const(x,K,M,P,rho,X,Y,T_P)
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
end
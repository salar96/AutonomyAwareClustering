function [Psi, f_val] = CrTempDA_Computational(X, K)
    M = size(X,1); N = 2; Tmin = 0.0005; alpha = 0.99; PERTURB = 0.0001; 
    STOP = 1e-2; T = 15; Px = (1/M)*ones(M,1); Y = repmat(Px'*X, [K,1]);
    while T >= Tmin
        L_old = inf;
        while 1
            [D,~] = distortion_DA(X,Y,M,N,K);
            num = exp(-D/T);
            den = repmat(sum(num,2),[1 K]);
            P = num./den;
            Py = P'*Px;
            Y = P'*(X.*repmat(Px,[1 N]))./repmat(Py,[1 N]) + PERTURB*rand(size(Y));
            L = -T*Px'*log(sum(exp(-D/T),2));
            if(norm(L-L_old) < STOP)
                break;
            end
            L_old = L;
        end
        Psi = zeros(size(Y));
        x_temp = Psi'; x0 = x_temp(:);
        fun = @(x)objective(x,K,M,P,Px,T,X,Y);
        nonclon = @(x)nonlin_const(x,K,M,P,Px,X,Y);
        A = []; b = []; Aeq = []; beq = []; lb = -ones(size(x0)); ub = ones(size(x0));
        options = optimoptions('fmincon','Display','off');
        [Psi, f_val] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonclon, options);
        fprintf('%d %d \n',T, f_val);
        T = T*alpha;
    end
end
function fun = objective(x,K,M,P,Px,T,X,Y)
    fun = 0; Py = P'*Px;
    x_temp = reshape(x,[2,K]); Psi = x_temp';
    Cov = cell(K,1);
    for j = 1 : K
        Cov{j} = zeros(2,2);
        for i = 1 : M
            Cov{j} = Cov{j} + (Px(i)*P(i,j)/Py(j))*(eye(2) - (2/T)*(X(i,:)-Y(j,:))'*(X(i,:)-Y(j,:)));
        end
        fun = fun + Py(j)*Psi(j,:)*Cov{j}*Psi(j,:)';
    end
end
function [c, c_eq] = nonlin_const(x,K,M,P,Px,X,Y)
    c_eq = zeros(2,1); c = [];
    x_temp = reshape(x,[2,K]); Psi = x_temp';
    Cov2 = {};
    for i = 1 : M
        Cov2{i} = 0;
        for j = 1 : K
            Cov2{i} = Cov2{i} + P(i,j)*Psi(j,:)*(X(i,:)-Y(j,:))';
        end
        c_eq(1) = c_eq(1) + Px(i)*Cov2{i}^2;
    end
    c_eq(2) = norm(Psi) - 1;
end
function T_cr = critical_betaDA(X, Y, M, P, rho)
    Px = rho; Py = P'*Px;
    C_ij = {}; eigMax = zeros(size(Y,1));
    for j = 1 : size(Y,1)
        C_ij{j} = zeros(2,2);
        for i = 1 : M
            C_ij{j} = C_ij{j} + P(i,j)*Px(i)/Py(j)*(X(i,:)-Y(j,:))'*(X(i,:)-Y(j,:));
        end
        eigMax(j) = max(eig(C_ij{j}));
    end
    [MaxEig, ~] = max(eigMax,[],'all');
    T_cr = 2*MaxEig;
end
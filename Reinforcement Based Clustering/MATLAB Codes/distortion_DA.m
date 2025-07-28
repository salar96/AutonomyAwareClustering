function [Du, D] = distortion_DA(X,Y,M,N,K)
    X = X'; X = X(:); X = X';
    Xar = repmat(X,[K 1]);
    Yar = repmat(Y,[1 M]);
    D2 = (Xar - Yar).^2;
    
    D = zeros(K,M);
    for i = 1:N
        D = D + D2(:,i:N:end);
    end
    D = D';
    Dm = min(D')';
    Du = D;
    D = D - repmat(Dm,[1 K]);
end

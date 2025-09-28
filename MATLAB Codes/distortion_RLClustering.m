function [D_bar,Du] = distortion_RLClustering(X,Y,M,N,K,T_P)
    X = X'; X = X(:); X = X';
    Xar = repmat(X,[K 1]);
    Yar = repmat(Y,[1 M]);
    D2 = (Xar - Yar).^2;
    
    D = zeros(K,M);
    for i = 1:N
        D = D + D2(:,i:N:end);
    end
    
    T_P_perm = permute(T_P, [2, 3, 1]);  % Now: K (j) × M (i) × K (k)

    % Expand D to match T_P dimensions: K × M → 1 × M × K
    D_expand = permute(D, [3, 2, 1]);    % Now: 1 × M × K
    
    % Element-wise multiply and sum over k (3rd dim)
    D_bar = squeeze(sum(T_P_perm .* D_expand, 3));

    D_bar = D_bar';
    Dm = min(D_bar')';
    Du = D_bar;
    D_bar = D_bar - repmat(Dm,[1 K]);
end
